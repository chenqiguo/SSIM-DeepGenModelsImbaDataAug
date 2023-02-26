#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:14:29 2022

@author: guo.1648
"""

# this code is to find the original class labels for each class re-named by cGAN.

# referenced from /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/cls_res_test.py

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
parser.add_argument('--test-data', metavar='DIR',
                    help='path to testing dataset')
parser.add_argument('--test-result', metavar='DIR',
                    help='path to testing results')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--inChs', default=3, type=int, metavar='C',
                    help='number of input channels')
parser.add_argument('--network', default='', type=str, metavar='PATH',
                    help='path to saved training model checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
#parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                    help='number of total epochs to run')
#parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
#parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                    metavar='LR', help='initial learning rate', dest='lr')
#parser.add_argument('--everyEpoch', default=30, type=int, metavar='E',
#                    help='used in func adjust_learning_rate')
#parser.add_argument('--saveEveryEpoch', default=0, type=int, metavar='sE',
#                    help='save the checkpoint for every sE epoch; 0 means NOT save')
#parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                    help='momentum')
#parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                    metavar='W', help='weight decay (default: 1e-4)',
#                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
#parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                    help='evaluate model on validation set')
#parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')
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
target_mapto_cGAN_dict = {}

# get from cls_res_test.py; also copied to corresponding GANinversion code:


"""
## for Insects:
pred_mapto_actual_dict = {0: '100', 1: '101', 2: '102', 3: '103', 4: '104', 5: '105',
                          6: '106', 7: '107', 8: '108', 9: '109', 10: '110', 11: '111',
                          12: '112', 13: '113', 14: '114', 15: '115', 16: '116', 17: '117',
                          18: '118', 19: '119', 20: '12', 21: '120', 22: '121', 23: '122',
                          24: '123', 25: '124', 26: '125', 27: '126', 28: '127', 29: '128',
                          30: '129', 31: '13', 32: '130', 33: '131', 34: '132', 35: '133',
                          36: '134', 37: '135', 38: '136', 39: '137', 40: '138', 41: '139',
                          42: '14', 43: '140', 44: '141', 45: '142', 46: '143', 47: '144',
                          48: '145', 49: '146', 50: '147', 51: '148', 52: '149', 53: '15',
                          54: '150', 55: '151', 56: '152', 57: '16', 58: '17', 59: '18',
                          60: '19', 61: '20', 62: '21', 63: '22', 64: '23', 65: '24',
                          66: '25', 67: '26', 68: '27', 69: '28', 70: '29', 71: '30',
                          72: '31', 73: '32', 74: '33', 75: '34', 76: '35', 77: '36',
                          78: '37', 79: '38', 80: '39', 81: '40', 82: '41', 83: '42',
                          84: '43', 85: '44', 86: '45', 87: '46', 88: '47', 89: '48',
                          90: '49', 91: '50', 92: '51', 93: '52', 94: '53', 95: '54',
                          96: '55', 97: '56', 98: '57', 99: '58', 100: '59', 101: '60',
                          102: '61', 103: '62', 104: '63', 105: '64', 106: '65', 107: '66',
                          108: '67', 109: '68', 110: '69', 111: '70', 112: '71', 113: '72',
                          114: '73', 115: '74', 116: '75', 117: '76', 118: '77', 119: '78',
                          120: '79', 121: '80', 122: '81', 123: '82', 124: '83', 125: '84',
                          126: '85', 127: '86', 128: '87', 129: '88', 130: '89', 131: '90',
                          132: '91', 133: '92', 134: '93', 135: '94', 136: '95', 137: '96',
                          138: '97', 139: '98', 140: '99'}
"""
"""
## for Fungi:
pred_mapto_actual_dict = {0: '0', 1: '1', 2: '10', 3: '11', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9'}
"""
"""
## for Birds:
pred_mapto_actual_dict = {0: '202', 1: '203', 2: '204', 3: '205', 4: '206', 5: '207',
                          6: '208', 7: '209', 8: '210', 9: '211', 10: '212', 11: '213',
                          12: '214', 13: '215', 14: '216', 15: '217', 16: '218', 17: '219',
                          18: '220', 19: '221', 20: '222', 21: '223', 22: '224', 23: '225',
                          24: '226', 25: '227', 26: '228', 27: '229', 28: '230', 29: '231',
                          30: '232', 31: '233', 32: '234', 33: '235', 34: '236', 35: '237',
                          36: '238', 37: '239', 38: '240', 39: '241', 40: '242', 41: '243',
                          42: '244', 43: '245', 44: '246', 45: '247', 46: '248', 47: '249',
                          48: '250', 49: '251', 50: '252', 51: '253', 52: '254', 53: '255',
                          54: '256', 55: '257', 56: '258', 57: '259', 58: '260', 59: '261',
                          60: '262', 61: '263', 62: '264', 63: '265', 64: '266', 65: '267',
                          66: '268', 67: '269', 68: '270', 69: '271', 70: '272', 71: '273',
                          72: '274', 73: '275', 74: '276', 75: '277', 76: '278', 77: '279',
                          78: '280', 79: '281', 80: '282', 81: '283', 82: '284', 83: '285',
                          84: '286', 85: '287', 86: '288', 87: '289', 88: '290', 89: '291',
                          90: '292', 91: '293', 92: '294', 93: '295', 94: '296', 95: '297',
                          96: '298', 97: '299', 98: '300', 99: '301', 100: '302', 101: '303',
                          102: '304', 103: '305', 104: '306', 105: '307', 106: '308', 107: '309',
                          108: '310', 109: '311', 110: '312', 111: '313', 112: '314', 113: '315',
                          114: '316', 115: '317', 116: '318', 117: '319', 118: '320', 119: '321',
                          120: '322', 121: '323', 122: '324', 123: '325', 124: '326', 125: '327'}
"""
"""
## for Reptiles:
pred_mapto_actual_dict = {0: '163', 1: '164', 2: '165', 3: '166', 4: '167', 5: '168', 6: '169',
                            7: '170', 8: '171', 9: '172', 10: '173', 11: '174', 12: '175', 13: '176',
                            14: '177', 15: '178', 16: '179', 17: '180', 18: '181', 19: '182', 20: '183',
                            21: '184', 22: '185', 23: '186', 24: '187', 25: '188', 26: '189', 27: '190',
                            28: '191', 29: '192', 30: '193', 31: '194', 32: '195', 33: '196', 34: '197',
                            35: '198', 36: '199', 37: '200', 38: '201'}
"""
"""
## for Amphibians:
pred_mapto_actual_dict = {0: '153', 1: '154', 2: '155', 3: '156', 4: '157', 5: '158', 6: '159', 7: '160', 8: '161', 9: '162'}
"""
"""
## for flowers:
pred_mapto_actual_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
"""
"""
## for UTKFace:
pred_mapto_actual_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
"""

## for scene:
pred_mapto_actual_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}





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
    global target_mapto_cGAN_dict
    
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
    """
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
    """
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    if args.inChs == 4 and args.arch == 'resnet18': # for RGB-concate-psketch
        print('HERE2') # just for debug!
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
    if 'iNaturalist_allSubCls' in args.test_data and args.arch == 'resnet18': # for class num > 1000
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
    """
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    """
    
    # newly modified by Chenqi for the testing:
    assert(args.network)
    assert(os.path.isfile(args.network))
    print("=> loading model checkpoint '{}'".format(args.network))
    if args.gpu is None:
        checkpoint = torch.load(args.network)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.network, map_location=loc)
    #args.start_epoch = checkpoint['epoch']
    """
    # in fact, we do NOT need them:
    if 'best_acc1' in checkpoint.keys():
        best_acc1 = checkpoint['best_acc1']
    else:
        best_acc1 = checkpoint['acc1_val']
    
    if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
    """
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.network, checkpoint['epoch']))
    
    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    testdir = args.test_data
    assert(os.path.exists(testdir))
    
    # newly modified:
    #if 'iNaturalist' in args.test_data and 'Insects' in args.test_data:
    if 'Insects' in args.test_data:
        #"""
        print('--> For original iNaturalist Insects dataset')
        normalize = transforms.Normalize(mean=[0.489, 0.485, 0.350],
                                         std=[0.187, 0.181, 0.173])
        #"""
        """
        print('--> 1st run: For styleGAN2 iNaturalist Insects dataset v3-cls')
        normalize = transforms.Normalize(mean=[0.391, 0.458, 0.424],
                                         std=[0.154, 0.150, 0.153])
        """
        """
        print('--> 2nd / 3rd run: For styleGAN2 iNaturalist Insects dataset v3-cls')
        normalize = transforms.Normalize(mean=[0.391, 0.463, 0.436],
                                         std=[0.153, 0.150, 0.153])
        """
    
    if 'Birds' in args.test_data:
        print('--> For original iNaturalist Birds dataset')
        normalize = transforms.Normalize(mean=[0.499, 0.520, 0.480],
                                         std=[0.162, 0.166, 0.182]) # from get_mean_std.py
        
    if 'Reptiles' in args.test_data:
        print('--> For original iNaturalist Reptiles dataset')
        normalize = transforms.Normalize(mean=[0.506, 0.471, 0.414],
                                         std=[0.191, 0.184, 0.177]) # from get_mean_std.py
    
    if 'Fungi' in args.test_data:
        #"""
        print('--> For original iNaturalist Fungi dataset')
        normalize = transforms.Normalize(mean=[0.427, 0.382, 0.318],
                                         std=[0.243, 0.225, 0.212]) # from get_mean_std.py
        #"""
        """
        print('--> 2nd run: For styleGAN2 iNaturalist Fungi dataset v3-cls')
        ### Fungi_forDebug:
        normalize = transforms.Normalize(mean=[0.316, 0.361, 0.394],
                                         std=[0.190, 0.190, 0.203]) # from get_mean_std.py
        """
      
    if 'Amphibians' in args.test_data:
        """
        print('--> For original iNaturalist Amphibians dataset')
        normalize = transforms.Normalize(mean=[0.475, 0.452, 0.372],
                                         std=[0.194, 0.190, 0.180]) # from get_mean_std.py
        """
        print('--> For styleGAN2 iNaturalist Amphibians dataset')
        normalize = transforms.Normalize(mean=[0.469, 0.447, 0.364],
                                         std=[0.180, 0.175, 0.164]) # from get_mean_std.py
    
    if 'flowers' in args.test_data:
        print('--> For orig flowers dataset')
        normalize = transforms.Normalize(mean=[0.497, 0.439, 0.311],
                                         std=[0.234, 0.209, 0.209]) # from get_mean_std.py
 
    
    if 'UTKFace' in args.test_data:
        print('--> For orig UTKFace dataset')
        normalize = transforms.Normalize(mean=[0.637, 0.478, 0.407],
                                         std=[0.182, 0.165, 0.156]) # from get_mean_std.py
    
    if 'scene' in args.test_data:
        print('--> For orig scene dataset')
        normalize = transforms.Normalize(mean=[0.438, 0.463, 0.456],
                                         std=[0.208, 0.203, 0.206]) # from get_mean_std.py
    
    
    
    #print('HERE3!!!')
    #assert(False)
    """
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
    
    """
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    """
    """
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    """
    """
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
    """
    
    print('=> loading test data')
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # test on the testing set and save the image names that are classified to each class:
    # note: keys are imgcGANName, items are predicted target index instead of actual cls index!!!
    results_tmp = my_test_func(test_loader, model, criterion, args) # need to map the predicted target index into actual cls index!
    
    """
    # just for debug:
    print('$$$$$$$$$$$$$$ DEBUG4!!! $$$$$$$$$$$$$$')
    #print('results = ' + str(results))
    print('target_mapto_cGAN_dict = ' + str(target_mapto_cGAN_dict))
    #assert(False)
    """
    
    #print('$$$$$$$$$$$$$$ DEBUG!!! $$$$$$$$$$$$$$')
    #print('target_mapto_cGAN_dict = ' + str(target_mapto_cGAN_dict))
    """
    f_pkl = open(args.test_result+'/results_tmp.pkl', 'wb')
    pickle.dump(results_tmp,f_pkl)
    f_pkl.close()
    """
    
    # map the predicted target index into actual cls index!
    #"""
    results = {}
    for imgcGANName in results_tmp:
        tmp_cls = results_tmp[imgcGANName]
        actual_cls = pred_mapto_actual_dict[tmp_cls]
        results[imgcGANName] = actual_cls
    
    f_pkl = open(args.test_result+'/results.pkl', 'wb')
    pickle.dump(results,f_pkl)
    f_pkl.close()
    
    #"""
    
    return


def my_test_func(test_loader, model, criterion, args):
    # referenced from validate() func
    
    print('=> testing...')
    batch_time = AverageMeter('Time', ':6.3f')
    #losses = AverageMeter('Loss', ':.4e')
    #top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time], #, losses, top1, top5],
        prefix='Test: ')
    
    results_tmp = {}
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            #"""
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            #"""
            """
            print('$$$$$$$$$$$$$$ DEBUG0!!! $$$$$$$$$$$$$$')
            sample_fname, _ = test_loader.dataset.samples[i]
            print('sample_fname = ' + str(sample_fname))
            print('test_loader.dataset.samples = ' + str(test_loader.dataset.samples))
            """
            # newly added by Chenqi: get the target_mapto_cGAN_dict:
            # a dict with system assigned target index as keys, and actual cGANimg name as items:
            this_testLoader_samples = test_loader.dataset.samples[i*args.batch_size:(i+1)*args.batch_size]
            assert(len(this_testLoader_samples) == len(target))
            
            imgcGANNames_list = []
            for j, sample_fname_pair in enumerate(this_testLoader_samples):
                target_j = target[j].item()
                #print('************')
                #print('target_j = ' + str(target_j))
                #print('sample_fname_pair = ' + str(sample_fname_pair))
                #actualCls_j = sample_fname_pair[0].split('/')[-2]
                cGAN_imgKey = sample_fname_pair[0].split('_128/')[-1]
                imgcGANNames_list.append(cGAN_imgKey)
                #print('actualCls_j = ' + str(actualCls_j))
                #assert(False) # just for debug!!!
                if target_j not in target_mapto_cGAN_dict:
                    target_mapto_cGAN_dict[target_j] = cGAN_imgKey
                    #print(target_mapto_cGAN_dict)
                    #assert(False) # just for debug!!!
            
            #print('target_mapto_cGAN_dict = ' + str(target_mapto_cGAN_dict)) # just for debug!
            
            # compute output
            output = model(images)
            """
            print('$$$$$$$$$$$$$$ DEBUG1!!! $$$$$$$$$$$$$$')
            print('output.shape = ' + str(output.shape))
            #print('output = ' + str(output))
            print('target.shape = ' + str(target.shape))
            print('target = ' + str(target))
            #assert(False)
            """
            
            # newly added by Chenqi:
            pred = get_imgPredCls(output, topk=(1,))
            pred = pred[0].tolist()
            """
            print('$$$$$$$$$$$$$$ DEBUG3!!! $$$$$$$$$$$$$$')
            print('len(pred) = ' + str(len(pred)))
            print('pred = ' + str(pred))
            assert(False)
            """
            
            assert(len(pred) == len(imgcGANNames_list))
            for k in range(len(pred)):
                this_imgcGANName = imgcGANNames_list[k]
                #this_actualCls = target_mapto_cGAN_dict[pred[k]]
                results_tmp[this_imgcGANName] = pred[k]
                #print('************')
                #print('results = ' + str(results))
                #assert(False) # just for debug!!!
            
            
            """
            ### we do NOT need these metrics since we are now doing the testing,
            ### and those testing labels (of GAN-syn imgs) are just garbage!
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            """
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        """
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        """

    return results_tmp


def get_imgPredCls(output, topk=(1,)):
    # referenced from func accuracy()
    """Computes the class prediction over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        #batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        """
        print('$$$$$$$$$$$$$$ DEBUG2!!! $$$$$$$$$$$$$$')
        print('pred.shape = ' + str(pred.shape)) # torch.Size([1, 256])
        print('pred = ' + str(pred))
        assert(False)
        """
        
        #correct = pred.eq(target.view(1, -1).expand_as(pred))

        #res = []
        #for k in topk:
        #    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            
        """
        print('****debug:')
        print('topk = ' + str(topk))
        print('correct_k = ' + str(correct_k))
        print('res = ' + str(res))
        assert(False)
        """
            
        return pred



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







if __name__ == '__main__':
    main()



