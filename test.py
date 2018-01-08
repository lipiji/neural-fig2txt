
import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io


def load():
    imgs = json.load(open("./data/dataset_coco.json", 'r'))
    
    #print imgs['dataset']
    imgs = imgs['images']
    
    for img in imgs:
        print img

load()

