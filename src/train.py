from __future__ import division
from zalo_utils import *
from sklearn.model_selection import train_test_split
# from utils import *
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
# import time
import torch.backends.cudnn as cudnn
import os, sys
from time import time, strftime

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices = [18, 34, 50, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type = str, help = 'optimizer')
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--num_epochs', default=1500, type=int,
                    help='Number of epochs in training')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--check_after', default=2,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1, 2],  # 0: from scratch, 1: from pretrained Resnet, 2: specific checkpoint in model_path
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")
parser.add_argument('--frozen_until', '-fu', type=int, default = 8,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float, 
        help = "number of training samples per class")
args = parser.parse_args()

KTOP = 3 # top k error

def exp_lr_scheduler(args, optimizer, epoch):
    # after epoch 100, not more learning rate decay
    init_lr = args.lr
    lr_decay_epoch = 4 # decay lr after each 10 epoch
    weight_decay = args.weight_decay
    lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch)) 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr


use_gpu = torch.cuda.is_available()

json_file = '../data/train_val2018.json'
data_dir = '../data/TrainVal/'

print('Loading data')
fns, lbs, cnt = get_fns_lbs(data_dir, json_file)
    
print('Total files in the original dataset: {}'.format(cnt))
print('Total files with > 0 byes: {}'.format(len(fns)))
print('Total files with zero bytes {}'.format(cnt - len(fns)))

############################333
print('Split data')
train_fns, val_fns, train_lbs, val_lbs = train_test_split(fns, lbs, test_size=args.val_ratio, random_state=2)
print('Number of training imgs: {}'.format(len(train_fns)))
print('Number of validation imgs: {}'.format(len(val_fns)))

########### 
print('DataLoader ....')
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
# input_size = 224
input_size = 224 

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        transforms.Scale(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

dsets = dict()
dsets['train'] = MyDataset(train_fns, train_lbs, transform=data_transforms['train'])
dsets['val'] = MyDataset(val_fns, val_lbs, transform=data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=args.batch_size,
                                   shuffle=(x != 'val'),
                                   num_workers=args.num_workers)
    for x in ['train', 'val']
}
########## 
print('Load model')

saved_model_fn = 'resnet' + '-%s' % (args.depth) + '_' + strftime('%m%d_%H%M')
old_model = './checkpoint/' + 'resnet' + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
if args.train_from == 2 and os.path.isfile(old_model):
    print("| Load pretrained at  %s..." % old_model)
    checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
    tmp = checkpoint['model']
    model = unparallelize_model(tmp)
    best_top3 = checkpoint['top3']
    print('previous top3\t%.4f'% best_top3)
    print('=============================================')
else:
    model = MyResNet(args.depth, len(set(train_lbs)))

##################
print('Start training ... ')
criterion = nn.CrossEntropyLoss()
model, optimizer = net_frozen(args, model)
model = parallelize_model(model)

N_train = len(train_lbs)
N_valid = len(val_lbs)
best_top3 = 1 
t0 = time()
for epoch in range(args.num_epochs):
    optimizer, lr = exp_lr_scheduler(args, optimizer, epoch) 
    print('#################################################################')
    print('=> Training Epoch #%d, LR=%.10f' % (epoch + 1, lr))
    # torch.set_grad_enabled(True)

    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    running_loss_src, running_corrects_src, tot_src = 0.0, 0.0, 0.0
    runnning_topk_corrects = 0.0
    ########################
    model.train()
    torch.set_grad_enabled(True)
    ## Training 
    # local_src_data = None
    for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['train']):
        optimizer.zero_grad()
        inputs = cvt_to_gpu(inputs)
        labels = cvt_to_gpu(labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss*inputs.shape[0]
        loss.backward()
        optimizer.step()
        ############################################
        _, preds = torch.max(outputs.data, 1)
        # topk 
        top3correct, _ = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
        runnning_topk_corrects += top3correct
        # pdb.set_trace()
        running_loss += loss.item()
        running_corrects += preds.eq(labels.data).cpu().sum()
        tot += labels.size(0)
        sys.stdout.write('\r')
        try:
            batch_loss = loss.item()
        except NameError:
            batch_loss = 0

        top1error = 1 - float(running_corrects)/tot
        top3error = 1 - float(runnning_topk_corrects)/tot
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f'
                         % (epoch + 1, args.num_epochs, batch_idx + 1,
                            (len(train_fns) // args.batch_size), batch_loss/args.batch_size,
                            top1error, top3error))
        sys.stdout.flush()
        sys.stdout.write('\r')

    top1error = 1 - float(running_corrects)/N_train
    top3error = 1 - float(runnning_topk_corrects)/N_train
    epoch_loss = running_loss/N_train
    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
            % (epoch_loss, top1error, top3error))

    print_eta(t0, epoch, args.num_epochs)

    ###################################
    ## Validation
    if (epoch + 1) % args.check_after == 0:
        # Validation 
        running_loss, running_corrects, tot = 0.0, 0.0, 0.0
        runnning_topk_corrects = 0
        torch.set_grad_enabled(False)
        model.eval()
        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
            inputs = cvt_to_gpu(inputs)
            labels = cvt_to_gpu(labels)
            outputs = model(inputs)
            _, preds  = torch.max(outputs.data, 1)
            top3correct, top3error = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
            runnning_topk_corrects += top3correct
            running_loss += loss.item()
            running_corrects += preds.eq(labels.data).cpu().sum()
            tot += labels.size(0)

        epoch_loss = running_loss / N_valid 
        top1error = 1 - float(running_corrects)/N_valid
        top3error = 1 - float(runnning_topk_corrects)/N_valid
        print('| Validation loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
                % (epoch_loss, top1error, top3error))

        ################### save model based on best top3 error
        if top3error < best_top3:
            print('Saving model')
            best_top3 = top3error
            best_model = copy.deepcopy(model)
            state = {
                'model': best_model,
                'top3' : best_top3,
                'args': args
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point + saved_model_fn + '.t7')
            print('=======================================================================')
            print('model saved to %s' % (save_point + saved_model_fn + '.t7'))
