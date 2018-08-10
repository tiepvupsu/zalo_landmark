from __future__ import division
from zalo_utils import *
from sklearn.model_selection import train_test_split
from utils import *
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
# from nets import load_model, parallelize_model

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices = [18, 50, 152], type=int, help='depth of model')
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', type=str, default = 512)

parser.add_argument('--frozen_until', '-fu', type=int, default=7,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float, 
        help = "number of training samples per class")
args = parser.parse_args()

KTOP = 3 # top k error

data_dir = '../data/Public/'
fn_all = [data_dir + fn for fn in os.listdir(data_dir) if fn.endswith('.jpg')]
fns = []
for fn in fn_all:
    # filter dammaged images
    if os.path.getsize(fn) > 0:
        fns.append(fn)

print('Total provided files: {}'.format(len(fn_all)))
print('Total damaged files: {}'.format(len(fn_all) - len(fns)))

lbs = [-1]*len(fns)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

dsets = dict()
dsets['test'] = MyDataset(fns, lbs, transform=data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=args.batch_size,
                                   shuffle= False,
                                   num_workers=8)
    for x in ['test']
}

#################
# load model 
old_model = './checkpoint/' + 'resnet' + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
print("| Load pretrained at  %s..." % old_model)
checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
tmp = checkpoint['model']
# model = 
model = unparallelize_model(tmp)
model = parallelize_model(model)
best_top3 = checkpoint['top3']
print('previous top3\t%.4f'% best_top3)
print('=============================================')

res_fn = 'mysubmission.csv'
f = open(res_fn, 'w')
header = 'id,predicted\n'
f.write(header)
torch.set_grad_enabled(False)
model.eval()
k = 3 # top 3 erro
tot = 0 
for batch_idx, (inputs, labels, fns0) in enumerate(dset_loaders['test']):
    inputs = cvt_to_gpu(inputs)
    # labels = cvt_to_gpu(labels)
    #1 outputs = model.predict(inputs, k=3)
    outputs = model(inputs)
    outputs = outputs.data.cpu().numpy()
    # pdb.set_trace()
    # topk = np.argsort(outputs, axis = 1)[:, -k:][:, ::-1]
    outputs = np.argsort(outputs, axis = 1)[:, -k:][:, ::-1]
    ########
    # write to file
    for i in range(len(fns0)):
        idx = fns0[i].split('/')[-1][:-4]
        # print(idx)
        tmp = idx + ','+ str(outputs[i, 0] )+ ' ' + str(outputs[i, 1]) + ' ' + str(outputs[i, 2]) +  '\n'
        f.write(tmp)
    tot += len(fns0)
    print('processed {}/{}'.format(tot, len(fns)))

f.close()
print('done')
