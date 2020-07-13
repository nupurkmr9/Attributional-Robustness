# -*- coding: utf-8 -*-
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.utils as vutils

from network import resnet
from tensorboardX import SummaryWriter
from utils_art.util_args import get_args
from utils_art.util_acc import accuracy, adjust_learning_rate, \
    save_checkpoint, AverageEpochMeter, SumEpochMeter, \
    ProgressEpochMeter, calculate_IOU, Logger
from utils_art.util_loader import data_loader
from utils_art.util_bbox import *
from utils_art.util_cam import *

def load_model(model, optimizer, args):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer

