#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:45:39 2021

@author: guo.1648
"""

# Classification on original/cartoonGAN imageNet with resnet50/18

import argparse
import os
import random
import shutil
import time
import warnings
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PIL import Image


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--result', metavar='DIR',
                    help='path to results')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--inChs', default=3, type=int, metavar='C',
                    help='number of input channels')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--everyEpoch', default=30, type=int, metavar='E',
                    help='used in func adjust_learning_rate')
parser.add_argument('--saveEveryEpoch', default=0, type=int, metavar='sE',
                    help='save the checkpoint for every sE epoch; 0 means NOT save')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#parser.add_argument('--world-size', default=-1, type=int,
#                    help='number of nodes for distributed training')
#parser.add_argument('--rank', default=-1, type=int,
#                    help='node rank for distributed training')
#parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                    help='url used to set up distributed training')
#parser.add_argument('--dist-backend', default='nccl', type=str,
#                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
#parser.add_argument('--multiprocessing-distributed', action='store_true',
#                    help='Use multi-processing distributed training to launch '
#                         'N processes per node, which has N GPUs. This is the '
#                         'fastest way to use PyTorch for either single node or '
#                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()
    
    """
    # just for debug:
    print('args.data = ' + str(args.data)) # /eecf/cbcsl/data100b/Chenqi/imageNet/data
    print('args.evaluate = ' + str(args.evaluate)) # False
    print('args.seed = ' + str(args.seed)) # None
    print('args.gpu = ' + str(args.gpu)) # 0
    assert(False)
    """

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    """
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    """
    
    ngpus_per_node = torch.cuda.device_count() # 2
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def my_4ch_loader(filename):
    #return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return Image.open(filename)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    """
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    """
    # create model
    # modified by Chenqi for 4-ch imgs:
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True) # returns a model pre-trained on ImageNet
        #assert(False)
        if args.inChs == 4 and args.arch == 'resnet18': # for RGB-concate-psketch
            print('HERE1') # just for debug!
            weight = model.conv1.weight.clone() # Copy the model weight
            model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
            model.conv1.weight[:, :3] = weight
            model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
    else: 
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        if args.inChs == 4 and args.arch == 'resnet18': # for RGB-concate-psketch
            print('HERE2') # just for debug!
            model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
        if 'iNaturalist_allSubCls' in args.data and args.arch == 'resnet18': # for class num > 1000
            print('HERE3') # just for debug!
            model.fc = nn.Linear(512, 1010)
        
    """
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    """
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        #print('HERE1!!!!')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        #print('HERE2!!!!')
        #assert(False)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            
            # newly modified by Chenqi:
            if 'best_acc1' in checkpoint.keys():
                best_acc1 = checkpoint['best_acc1']
            else:
                best_acc1 = checkpoint['acc1_val']
            
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    # newly modified:
    if 'iNaturalist' in args.data and 'Insects' in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For original iNaturalist Insects dataset')
            normalize = transforms.Normalize(mean=[0.489, 0.485, 0.350],
                                             std=[0.187, 0.181, 0.173])
        
        elif 'cGAN' in args.data: # for the cGAN augmented dataset:
            if 'opt1' in args.data:
                print('--> For cGAN iNaturalist Insects dataset opt1 (random)')
                normalize = transforms.Normalize(mean=[0.493, 0.484, 0.349],
                                                 std=[0.182, 0.175, 0.165]) # from get_mean_std.py
            elif 'opt2/step1/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Insects dataset opt2 step1 thresh10')
                normalize = transforms.Normalize(mean=[0.492, 0.484, 0.346],
                                                 std=[0.184, 0.178, 0.167]) # from get_mean_std.py
            elif 'opt2/step2/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Insects dataset opt2 step2 thresh10')
                normalize = transforms.Normalize(mean=[0.492, 0.484, 0.346],
                                                 std=[0.185, 0.178, 0.167]) # from get_mean_std.py
            elif 'opt2/step2/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Insects dataset opt2 step2 thresh15')
                normalize = transforms.Normalize(mean=[0.492, 0.484, 0.346],
                                                 std=[0.184, 0.178, 0.167]) # from get_mean_std.py
             
        elif 'aug_data_v2' in args.data: # for the styleGAN2 augmented dataset (aug_data v2):
            print('--> For styleGAN2 iNaturalist Insects dataset v2')
            normalize = transforms.Normalize(mean=[0.396, 0.473, 0.445],
                                             std=[0.126, 0.125, 0.125]) # from get_mean_std.py
        elif 'aug_data_v3_cls' in args.data: # for the styleGAN2 augmented dataset (aug_data v3 cls):
            print('--> For styleGAN2 iNaturalist Insects dataset v3-cls')
            """
            print('Insects_step1:')
            normalize = transforms.Normalize(mean=[0.391, 0.458, 0.424],
                                             std=[0.154, 0.150, 0.153]) # from get_mean_std.py
            """
            """
            print('Insects_step2 step3:')
            normalize = transforms.Normalize(mean=[0.391, 0.463, 0.436],
                                             std=[0.153, 0.150, 0.153]) # from get_mean_std.py
            """
            """
            print('Insects_step4_v2 thresh_20:')
            normalize = transforms.Normalize(mean=[0.387, 0.465, 0.443],
                                             std=[0.151, 0.148, 0.152]) # from get_mean_std.py
            """
            print('Insects_step4_v2 thresh_25:')
            normalize = transforms.Normalize(mean=[0.386, 0.464, 0.444],
                                             std=[0.153, 0.150, 0.154]) # from get_mean_std.py
            
        elif 'aug_data_v3' in args.data: # for the styleGAN2 augmented dataset (aug_data v3):
            print('--> For styleGAN2 iNaturalist Insects dataset v3')
            normalize = transforms.Normalize(mean=[0.393, 0.460, 0.428],
                                             std=[0.137, 0.134, 0.135]) # from get_mean_std.py
        else: # for the styleGAN2 augmented dataset (aug_data v1):
            print('--> For styleGAN2 iNaturalist Insects dataset v1')
            normalize = transforms.Normalize(mean=[0.489, 0.482, 0.351],
                                             std=[0.182, 0.176, 0.169]) # from get_mean_std.py
    
    if 'iNaturalist' in args.data and 'Birds' in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For original iNaturalist Birds dataset')
            normalize = transforms.Normalize(mean=[0.499, 0.520, 0.480],
                                             std=[0.162, 0.166, 0.182]) # from get_mean_std.py
        elif 'cGAN' in args.data: # for the cGAN augmented dataset:
            if 'opt2/step1/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Birds dataset opt2 step1 thresh10')
                normalize = transforms.Normalize(mean=[0.493, 0.515, 0.474],
                                                 std=[0.156, 0.161, 0.177]) # from get_mean_std.py
            if 'opt2/step1/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Birds dataset opt2 step1 thresh15')
                normalize = transforms.Normalize(mean=[0.493, 0.514, 0.471],
                                                 std=[0.156, 0.161, 0.178]) # from get_mean_std.py
            if 'opt2/step2/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Birds dataset opt2 step2 thresh15')
                normalize = transforms.Normalize(mean=[0.493, 0.515, 0.473],
                                                 std=[0.156, 0.161, 0.177]) # from get_mean_std.py
            if 'opt2/step2/thresh_20' in args.data:
                print('--> For cGAN iNaturalist Birds dataset opt2 step2 thresh20')
                normalize = transforms.Normalize(mean=[0.492, 0.513, 0.471],
                                                 std=[0.157, 0.162, 0.178]) # from get_mean_std.py
            if 'opt2/step2/thresh_25' in args.data:
                print('--> For cGAN iNaturalist Birds dataset opt2 step2 thresh25')
                normalize = transforms.Normalize(mean=[0.490, 0.510, 0.466],
                                                 std=[0.159, 0.164, 0.180]) # from get_mean_std.py
                
        else: # for the styleGAN2 augmented dataset:
            print('--> For styleGAN2 iNaturalist Birds dataset')
            normalize = transforms.Normalize(mean=[0.495, 0.516, 0.474],
                                             std=[0.162, 0.168, 0.185]) # from get_mean_std.py
            
    if 'iNaturalist' in args.data and 'Reptiles' in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For original iNaturalist Reptiles dataset')
            normalize = transforms.Normalize(mean=[0.506, 0.471, 0.414],
                                             std=[0.191, 0.184, 0.177]) # from get_mean_std.py
        elif 'cGAN' in args.data: # for the cGAN augmented dataset:
            if 'opt2/step1/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Reptiles dataset opt2 step1 thresh10')
                normalize = transforms.Normalize(mean=[0.509, 0.473, 0.414],
                                                 std=[0.176, 0.170, 0.162]) # from get_mean_std.py
            if 'opt2/step1/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Reptiles dataset opt2 step1 thresh15')
                normalize = transforms.Normalize(mean=[0.513, 0.478, 0.420],
                                                 std=[0.176, 0.170, 0.163]) # from get_mean_std.py
            if 'opt2/step2/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Reptiles dataset opt2 step2 thresh15')
                normalize = transforms.Normalize(mean=[0.510, 0.473, 0.414],
                                                 std=[0.176, 0.170, 0.162]) # from get_mean_std.py
            if 'opt2/step2/thresh_20' in args.data:
                print('--> For cGAN iNaturalist Reptiles dataset opt2 step2 thresh20')
                normalize = transforms.Normalize(mean=[0.511, 0.474, 0.415],
                                                 std=[0.176, 0.170, 0.162]) # from get_mean_std.py
                
        else: # for the styleGAN2 augmented dataset:
            print('--> For styleGAN2 iNaturalist Reptiles dataset')
            normalize = transforms.Normalize(mean=[0.504, 0.466, 0.410],
                                             std=[0.182, 0.174, 0.167]) # from get_mean_std.py
        
    if 'iNaturalist' in args.data and 'Fungi' in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For original iNaturalist Fungi dataset')
            normalize = transforms.Normalize(mean=[0.427, 0.382, 0.318],
                                             std=[0.243, 0.225, 0.212]) # from get_mean_std.py
        
        elif 'cGAN' in args.data: # for the cGAN augmented dataset:
            """
            if 'opt1' in args.data:
                print('--> For cGAN iNaturalist Fungi dataset opt1 (random)')
                normalize = transforms.Normalize(mean=[0.493, 0.484, 0.349],
                                                 std=[0.182, 0.175, 0.165]) # from get_mean_std.py
            """
            if 'opt2/step1/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Fungi dataset opt2 step1 thresh10')
                normalize = transforms.Normalize(mean=[0.452, 0.407, 0.329],
                                                 std=[0.242, 0.226, 0.212]) # from get_mean_std.py
            if 'opt2/step1/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Fungi dataset opt2 step1 thresh15')
                normalize = transforms.Normalize(mean=[0.463, 0.416, 0.333],
                                                 std=[0.241, 0.226, 0.211]) # from get_mean_std.py
            if 'opt2/step2/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Fungi dataset opt2 step2 thresh10')
                """
                ### based on step1 thresh10:
                normalize = transforms.Normalize(mean=[0.453, 0.408, 0.330],
                                                 std=[0.242, 0.227, 0.212]) # from get_mean_std.py
                """
                ### based on step1 thresh15:
                normalize = transforms.Normalize(mean=[0.450, 0.407, 0.329],
                                                 std=[0.242, 0.227, 0.212]) # from get_mean_std.py
                
            if 'opt2/step2/thresh_20' in args.data:
                print('--> For cGAN iNaturalist Fungi dataset opt2 step2 thresh20')
                """
                ### based on step1 thresh10:
                normalize = transforms.Normalize(mean=[0.453, 0.408, 0.329],
                                                 std=[0.242, 0.227, 0.212]) # from get_mean_std.py
                """
                ### based on step1 thresh15:
                normalize = transforms.Normalize(mean=[0.452, 0.408, 0.330],
                                                 std=[0.243, 0.227, 0.213]) # from get_mean_std.py
                
            if 'opt2/step2/thresh_30' in args.data:
                print('--> For cGAN iNaturalist Fungi dataset opt2 step2 thresh30')
                """
                ### based on step1 thresh10:
                normalize = transforms.Normalize(mean=[0.454, 0.409, 0.333],
                                                 std=[0.245, 0.229, 0.215]) # from get_mean_std.py
                """
                ### based on step1 thresh15:
                normalize = transforms.Normalize(mean=[0.459, 0.413, 0.337],
                                                 std=[0.246, 0.231, 0.216]) # from get_mean_std.py
            
            
        elif 'aug_data_v3_cls' in args.data: # for the styleGAN2 augmented dataset (aug_data v3 cls):
            print('--> For styleGAN2 iNaturalist Fungi dataset v3-cls')
            """
            ### Fungi_forDebug:
            normalize = transforms.Normalize(mean=[0.316, 0.361, 0.394],
                                             std=[0.190, 0.190, 0.203]) # from get_mean_std.py
            """
            """
            ### Fungi_forDebug2:
            normalize = transforms.Normalize(mean=[0.317, 0.362, 0.395],
                                             std=[0.189, 0.188, 0.202]) # from get_mean_std.py
            """
            """
            ### Fungi_forDebug3:
            normalize = transforms.Normalize(mean=[0.341, 0.361, 0.370],
                                             std=[0.195, 0.190, 0.198]) # from get_mean_std.py
            """
            """
            ### Fungi_forDebug_step2:
            normalize = transforms.Normalize(mean=[0.294, 0.340, 0.378],
                                             std=[0.183, 0.188, 0.205]) # from get_mean_std.py
            """
            """
            ### Fungi_forDebug3_step2:
            normalize = transforms.Normalize(mean=[0.384, 0.349, 0.304],
                                             std=[0.211, 0.193, 0.188]) # from get_mean_std.py
            """
            """
            print('Fungi_forDebug_step3:')
            normalize = transforms.Normalize(mean=[0.293, 0.339, 0.376],
                                             std=[0.184, 0.189, 0.207]) # from get_mean_std.py
            """
            """
            print('Fungi_forDebug3_step3:')
            normalize = transforms.Normalize(mean=[0.368, 0.338, 0.295],
                                             std=[0.204, 0.185, 0.183]) # from get_mean_std.py
            """
            """
            print('Fungi_forDebug_step4:')
            normalize = transforms.Normalize(mean=[0.290, 0.338, 0.376],
                                             std=[0.182, 0.188, 0.204]) # from get_mean_std.py
            """
            """
            if 'thresh_20' in args.data:
                print('Fungi_forDebug_step3_v2 thresh_20:')
                normalize = transforms.Normalize(mean=[0.293, 0.339, 0.375],
                                                 std=[0.186, 0.192, 0.209]) # from get_mean_std.py
            if 'thresh_30' in args.data:
                print('Fungi_forDebug_step3_v2 thresh_30:')
                normalize = transforms.Normalize(mean=[0.294, 0.339, 0.375],
                                                 std=[0.195, 0.200, 0.220]) # from get_mean_std.py
            if 'thresh_40' in args.data:
                print('Fungi_forDebug_step3_v2 thresh_40:')
                normalize = transforms.Normalize(mean=[0.300, 0.342, 0.380],
                                                 std=[0.202, 0.207, 0.228]) # from get_mean_std.py
            """
            if 'thresh_20' in args.data:
                print('Fungi_forDebug_step4_v2 thresh_20:')
                normalize = transforms.Normalize(mean=[0.292, 0.339, 0.375],
                                                 std=[0.192, 0.198, 0.215]) # from get_mean_std.py
            if 'thresh_30' in args.data:
                print('Fungi_forDebug_step4_v2 thresh_30:')
                normalize = transforms.Normalize(mean=[0.295, 0.341, 0.380],
                                                 std=[0.203, 0.211, 0.232]) # from get_mean_std.py
            if 'thresh_35' in args.data:
                print('Fungi_forDebug_step4_v2 thresh_35:')
                normalize = transforms.Normalize(mean=[0.295, 0.342, 0.384],
                                                 std=[0.208, 0.217, 0.240]) # from get_mean_std.py
            
            """
            ### Fungi_method_same_as_Insects:
            normalize = transforms.Normalize(mean=[0.288, 0.342, 0.382],
                                             std=[0.179, 0.185, 0.200]) # from get_mean_std.py
            """
        elif 'aug_data_v3' in args.data: # for the styleGAN2 augmented dataset (aug_data v3):
            print('--> For styleGAN2 iNaturalist Fungi dataset v3')
            normalize = transforms.Normalize(mean=[0.283, 0.338, 0.378],
                                             std=[0.172, 0.178, 0.193]) # from get_mean_std.py
        else: # for the styleGAN2 augmented dataset:
            print('--> For styleGAN2 iNaturalist Fungi dataset')
            normalize = transforms.Normalize(mean=[0.447, 0.393, 0.327],
                                             std=[0.241, 0.216, 0.202]) # from get_mean_std.py
    
    if 'iNaturalist' in args.data and 'Amphibians' in args.data and 'diffusion' not in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For original iNaturalist Amphibians dataset')
            normalize = transforms.Normalize(mean=[0.474, 0.451, 0.370],
                                             std=[0.194, 0.190, 0.180]) # from get_mean_std.py
        
        elif 'cGAN' in args.data: # for the cGAN augmented dataset:
            if 'opt2/step1/thresh_10' in args.data:
                print('--> For cGAN iNaturalist Amphibians dataset opt2 step1 thresh10')
                normalize = transforms.Normalize(mean=[0.462, 0.433, 0.343],
                                                 std=[0.186, 0.180, 0.167]) # from get_mean_std.py
            if 'opt2/step1/thresh_15' in args.data:
                print('--> For cGAN iNaturalist Amphibians dataset opt2 step1 thresh15')
                normalize = transforms.Normalize(mean=[0.455, 0.430, 0.344],
                                                 std=[0.190, 0.186, 0.175]) # from get_mean_std.py
            if 'opt2/step2/thresh_20' in args.data:
                print('--> For cGAN iNaturalist Amphibians dataset opt2 step2 thresh20')
                normalize = transforms.Normalize(mean=[0.462, 0.431, 0.341],
                                                 std=[0.186, 0.181, 0.167]) # from get_mean_std.py
            if 'opt2/step2/thresh_30' in args.data:
                print('--> For cGAN iNaturalist Amphibians dataset opt2 step2 thresh30')
                normalize = transforms.Normalize(mean=[0.463, 0.430, 0.339],
                                                 std=[0.190, 0.185, 0.171]) # from get_mean_std.py
            
        else: 
            """
            # for the styleGAN2 augmented dataset:
            print('--> For styleGAN2 iNaturalist Amphibians dataset')
            normalize = transforms.Normalize(mean=[0.469, 0.447, 0.364],
                                             std=[0.180, 0.175, 0.164]) # from get_mean_std.py
            """
            if 'step1_v2/thresh_12' in args.data:
                print('Amphibians_step1_v2 thresh_12:')
                normalize = transforms.Normalize(mean=[0.378, 0.432, 0.444],
                                                 std=[0.123, 0.122, 0.124]) # from get_mean_std.py
            
    if ('iNaturalist_allSubCls' in args.data) or ('iNaturalist_allSuperCls' in args.data):
        if 'aug' not in args.data: # for the original dataset:
            print('--> For original iNaturalist allSubCls / allSuperCls dataset')
            normalize = transforms.Normalize(mean=[0.460, 0.479, 0.368],
                                             std=[0.186, 0.184, 0.183]) # from get_mean_std.py
        else: # for the styleGAN2 augmented dataset:
            print('--> For styleGAN2 iNaturalist allSubCls / allSuperCls dataset')
            normalize = transforms.Normalize(mean=[0.473, 0.461, 0.376],
                                             std=[0.190, 0.184, 0.179]) # from get_mean_std.py
    
    
    if 'flowers' in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For orig flowers dataset')
            normalize = transforms.Normalize(mean=[0.497, 0.439, 0.311],
                                             std=[0.234, 0.209, 0.209]) # from get_mean_std.py
        else: # for the cGAN augmented dataset:
            if 'step1/thresh_15' in args.data:
                print('--> For cGAN-aug flowers dataset step1 thresh_15')
                normalize = transforms.Normalize(mean=[0.515, 0.462, 0.317],
                                                 std=[0.233, 0.208, 0.211]) # from get_mean_std.py
            if 'step2/thresh_20' in args.data:
                print('--> For cGAN-aug flowers dataset step2 thresh_20')
                normalize = transforms.Normalize(mean=[0.522, 0.450, 0.314],
                                                 std=[0.232, 0.205, 0.209]) # from get_mean_std.py
            
    if 'UTKFace' in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For orig UTKFace dataset')
            normalize = transforms.Normalize(mean=[0.637, 0.478, 0.407],
                                             std=[0.182, 0.165, 0.156]) # from get_mean_std.py
        else: # for the cGAN augmented dataset:
            if 'step1/thresh_25' in args.data:
                print('--> For cGAN-aug UTKFace dataset step1 thresh_25')
                normalize = transforms.Normalize(mean=[0.643, 0.480, 0.409],
                                                 std=[0.185, 0.168, 0.159]) # from get_mean_std.py
            if 'step2/thresh_25' in args.data:
                print('--> For cGAN-aug UTKFace dataset step2 thresh_25')
                normalize = transforms.Normalize(mean=[0.641, 0.477, 0.406],
                                                 std=[0.186, 0.168, 0.158]) # from get_mean_std.py
    
    if 'scene' in args.data and 'diffusion' not in args.data:
        if 'aug' not in args.data: # for the original dataset:
            print('--> For orig scene dataset')
            normalize = transforms.Normalize(mean=[0.438, 0.463, 0.456],
                                             std=[0.208, 0.203, 0.206]) # from get_mean_std.py
        else: # for the cGAN augmented dataset:
            if 'step1/thresh_30' in args.data:
                print('--> For cGAN-aug scene dataset step1 thresh_30')
                normalize = transforms.Normalize(mean=[0.437, 0.465, 0.457],
                                                 std=[0.206, 0.202, 0.205]) # from get_mean_std.py
            if 'step2/thresh_30' in args.data:
                print('--> For cGAN-aug scene dataset step2 thresh_30')
                normalize = transforms.Normalize(mean=[0.437, 0.464, 0.456],
                                                 std=[0.205, 0.201, 0.204]) # from get_mean_std.py
        
    
    if 'imbalanced_data/resnet/data/iNaturalist' in args.data and 'iNaturalist_' not in args.data:
        # for the original dataset:
        print('--> For orig iNaturalist (all but Plants) dataset')
        normalize = transforms.Normalize(mean=[0.494, 0.495, 0.416],
                                         std=[0.178, 0.176, 0.178]) # from get_mean_std.py
    elif 'imbalanced_data/resnet/aug_data_cGAN/iNaturalist_cGANaug' in args.data: 
        # for the cGAN augmented dataset:
        print('--> For cGAN-aug iNaturalist (all but Plants) dataset')
        normalize = transforms.Normalize(mean=[0.485, 0.478, 0.403],
                                         std=[0.179, 0.176, 0.174]) # from get_mean_std.py
    
    
    if 'without_cls_select' in args.data:
        if 'Amphibians' in args.data:
            """
            print('--> For without_cls_select (opt1) iNaturalist Amphibians dataset')
            normalize = transforms.Normalize(mean=[0.461, 0.435, 0.352],
                                             std=[0.182, 0.177, 0.163]) # from get_mean_std.py
            """
            print('--> For without_cls_select (opt1) iNaturalist Amphibians 2nd try dataset')
            normalize = transforms.Normalize(mean=[0.462, 0.436, 0.352],
                                             std=[0.182, 0.177, 0.163]) # from get_mean_std.py
            
            
        elif 'Fungi' in args.data:
            print('--> For without_cls_select (opt1) iNaturalist Fungi dataset')
            normalize = transforms.Normalize(mean=[0.440, 0.397, 0.330],
                                             std=[0.239, 0.228, 0.216]) # from get_mean_std.py
        elif 'Reptiles' in args.data:
            print('--> For without_cls_select (opt1) iNaturalist Reptiles dataset')
            normalize = transforms.Normalize(mean=[0.504, 0.467, 0.410],
                                             std=[0.176, 0.170, 0.163]) # from get_mean_std.py
        elif 'Birds' in args.data:
            print('--> For without_cls_select (opt1) iNaturalist Birds dataset')
            normalize = transforms.Normalize(mean=[0.494, 0.516, 0.476],
                                             std=[0.154, 0.159, 0.175]) # from get_mean_std.py
        elif 'Insects' in args.data:
            print('--> For without_cls_select (opt1) iNaturalist Insects dataset')
            normalize = transforms.Normalize(mean=[0.495, 0.484, 0.353],
                                             std=[0.181, 0.174, 0.165]) # from get_mean_std.py
        elif 'flowers' in args.data:
            print('--> For without_cls_select (opt1) flowers dataset')
            normalize = transforms.Normalize(mean=[0.511, 0.449, 0.308],
                                             std=[0.229, 0.202, 0.203]) # from get_mean_std.py
        elif 'scene' in args.data:
            print('--> For without_cls_select (opt1) scene dataset')
            normalize = transforms.Normalize(mean=[0.438, 0.463, 0.455],
                                             std=[0.206, 0.202, 0.204]) # from get_mean_std.py
        elif 'UTKFace' in args.data:
            print('--> For without_cls_select (opt1) UTKFace dataset')
            normalize = transforms.Normalize(mean=[0.629, 0.469, 0.397],
                                             std=[0.188, 0.168, 0.157]) # from get_mean_std.py
        
    
    if 'diffusion' in args.data:
        if 'Fungi' in args.data:
            print('--> For guided diffusion model iNaturalist Fungi dataset')
            """
            # imgs selected by res18 trained on orig dataset:
            normalize = transforms.Normalize(mean=[0.417, 0.373, 0.310],
                                             std=[0.172, 0.164, 0.153]) # from get_mean_std.py
            """
            # imgs selected by res18 trained on cGAN-aug s2t10 dataset:
            normalize = transforms.Normalize(mean=[0.401, 0.366, 0.316],
                                             std=[0.177, 0.173, 0.163]) # from get_mean_std.py
        
        elif 'Birds' in args.data:
            print('--> For guided diffusion model iNaturalist Birds dataset')
            normalize = transforms.Normalize(mean=[0.499, 0.517, 0.484],
                                             std=[0.148, 0.153, 0.168]) # from get_mean_std.py
            
        elif 'scene' in args.data:
            print('--> For guided diffusion model scene dataset')
            normalize = transforms.Normalize(mean=[0.434, 0.457, 0.449],
                                             std=[0.203, 0.199, 0.203]) # from get_mean_std.py
        
        elif 'Amphibians' in args.data:
            print('--> For guided diffusion model iNaturalist Amphibians dataset')
            normalize = transforms.Normalize(mean=[0.470, 0.443, 0.367],
                                             std=[0.175, 0.170, 0.159]) # from get_mean_std.py
        
    
    #print('HERE3!!!')
    #assert(False)
    
    print('=> loading train data')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    # added by Chenqi for 4-ch imgs:
    if args.inChs == 4:
        train_dataset = datasets.ImageFolder(
            root=traindir,
            loader=my_4ch_loader, # for 4ch input imgs!
            transform=transforms.Compose([
                      transforms.RandomResizedCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize,
            ]))
    
    
    """
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    """
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    print('=> loading val data')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # added by Chenqi for 4-ch imgs:
    if args.inChs == 4:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=valdir, 
                                 loader=my_4ch_loader, # for 4ch input imgs!
                                 transform=transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize,
                                           ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    history = [] # newly added by Chenqi
    
    for epoch in range(args.start_epoch, args.epochs):
        """
        if args.distributed:
            train_sampler.set_epoch(epoch)
        """
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1_train, acc5_train = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        
        # newly added by Chenqi: save metrics to history:
        this_result = {'epoch': epoch + 1,
                       'arch': args.arch,
                       'acc1_train': acc1_train,
                       'acc5_train': acc5_train,
                       'acc1_val': acc1,
                       'acc5_val': acc5}
        history.append(this_result)
        
        # newly added by Chenqi: save checkpoint every saveEveryEpoch:
        if (args.saveEveryEpoch != 0) and not ((epoch + 1) % args.saveEveryEpoch):
            this_state = {'epoch': epoch + 1,
                          'arch': args.arch,
                          'state_dict': model.state_dict(),
                          'acc1_val': acc1,
                          'acc5_val': acc5,
                          'optimizer' : optimizer.state_dict()}
            torch.save(this_state, args.result+'/checkpoint_Epoch'+str(epoch + 1)+'.pth.tar')
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_state = {'epoch': epoch + 1,
                          'arch': args.arch,
                          'state_dict': model.state_dict(),
                          'best_acc1': best_acc1,
                          'corresponding_acc5': acc5,
                          'optimizer' : optimizer.state_dict()}
            torch.save(best_state, args.result+'/checkpoint_bestAcc1.pth.tar')
        
        """
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        """
    
    # save training history to pk:
    f_pkl = open(args.result+'/history.pkl', 'wb')
    pickle.dump(history,f_pkl)
    f_pkl.close()
    
        

def train(train_loader, model, criterion, optimizer, epoch, args):
    print('=> training...')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        """
        # just for debug:
        this_trainLoader_samples = train_loader.dataset.samples[i*args.batch_size:(i+1)*args.batch_size]
        assert(len(this_trainLoader_samples) == len(target))
        print('this_trainLoader_samples = ' + str(this_trainLoader_samples))
        print('images.shape = ' + str(images.shape))
        """
        #print(i) # for debug
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        
        # for debug:
        print('*******************debug: output = ')
        print(output) # logits
        print(output.shape) # torch.Size([256, 1000])
        print('*******************debug: target = ')
        print(target) # labels (class index start from 0)
        print(target.shape) # torch.Size([256])
        #assert(False)
        
        loss = criterion(output, target)
        
        # for debug:
        print('*******************debug: loss = ')
        print(loss) # tensor(7.2433, device='cuda:0', grad_fn=<NllLossBackward>)
        print(loss.shape) # torch.Size([])
        assert(False)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        """
        # just for debug:
        #print(loss.item().shape)
        print('target.shape = ' + str(target.shape)) # 256
        print('output.shape = ' + str(output.shape)) # 256, 1000
        #print('max(target) = ' + str(max(target)))
        #print('min(target) = ' + str(min(target)))
        #print('target = ' + str(target))
        #print(target)
        #print('output[0,:] = ' + str(output[0,:]))
        print('acc1 = ' + str(acc1))
        #print('images.size = ' + str(images.size))
        assert(False)
        """
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        """
        # for debug:
        print('----------- debug:')
        print('acc1 = ' + str(acc1))
        print('acc5 = ' + str(acc5))
        print('top1 = ' + str(top1))
        print('top5 = ' + str(top5))
        assert(False)
        """

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    """
    # for debug:
    print('----------- debug:')
    print('top1 = ' + str(top1))
    print('top5 = ' + str(top5))
    assert(False)
    """
    return (top1.avg, top5.avg)
    

def validate(val_loader, model, criterion, args):
    print('=> validating...')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return (top1.avg, top5.avg)

"""
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every everyEpoch epochs"""
    print('=> adjusting lr')
    
    # just for debug!:
    print('args.everyEpoch = ' + str(args.everyEpoch))
    print('before adjusting: args.lr = ' + str(args.lr))
    print('epoch = ' + str(epoch))
    print('args.start_epoch = ' + str(args.start_epoch))
    print(epoch - args.start_epoch)
    
    lr = args.lr * (0.1 ** (epoch // args.everyEpoch)) # 30 # orig code
    """
    ### NOT use!!!:
    if args.resume: # code modified by Chenqi:
        lr = args.lr * (0.1 ** ( (epoch - args.start_epoch + 1) // args.everyEpoch)) # or (epoch - args.start_epoch) ???
    """
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # just for debug!:
    print('after adjusting: lr = ' + str(lr))
    print('param_group[lr] = ' + str(param_group['lr']))
    #assert(False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        """
        print('****debug:')
        print('topk = ' + str(topk))
        print('correct_k = ' + str(correct_k))
        print('res = ' + str(res))
        assert(False)
        """
            
        return res


if __name__ == '__main__':
    main()


