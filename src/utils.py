import torch
from torch.autograd import Variable

import numpy as np
import os
#import tensorflow as tf
# from keras.preprocessing import image
# from keras.applications.vgg19 import preprocess_input
from time import time, strftime
# from tensorflow_vgg import utils, vgg19, vgg16
import skimage
from skimage.color import rgb2gray, gray2rgb, rgba2rgb # from tf_utils import DataBatch

from random import shuffle

