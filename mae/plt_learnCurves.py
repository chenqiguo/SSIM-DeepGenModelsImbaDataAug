#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:28:30 2022

@author: guo.1648
"""

# plot learning curves for MAE

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/'
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/results/cls_mae/'

supCls_folder = 'iNatruarlist_all_except_Plants/'

foldID_orig = '27810/'
foldID_cGAN = '4664/'

txtFile = 'log.txt'

model_arch = 'vit_huge_finetune_'



if __name__ == '__main__':
    
    orig_folder = srcRootDir + foldID_orig
    assert(os.path.exists(orig_folder))
    gan_folder = srcRootDir + foldID_cGAN
    assert(os.path.exists(gan_folder))
    
    epochs_orig = []
    valid_acc1_list_orig = []
    
    epochs_gan = []
    valid_acc1_list_gan= []
    
    # (1) for original images:
    orig_txt_fullName = orig_folder + txtFile
    assert(os.path.exists(orig_txt_fullName))
    
    with open(orig_txt_fullName) as file:
        for line in file:
            this_line = line.rstrip()
            
            if 'test_acc1' in this_line:
                this_epoch = this_line.split('\"epoch\": ')[-1].split(',')[0]
                this_epoch = int(this_epoch) #+ 1
                epochs_orig.append(this_epoch)
                this_val_acc1 = this_line.split('\"test_acc1\": ')[-1].split(',')[0]
                this_val_acc1 = float(this_val_acc1)
                valid_acc1_list_orig.append(this_val_acc1)
                
    # (2) for cGAN synthesized images:
    gan_txt_fullName = gan_folder + txtFile
    assert(os.path.exists(gan_txt_fullName))
    
    with open(gan_txt_fullName) as file:
        for line in file:
            this_line = line.rstrip()
            
            if 'test_acc1' in this_line:
                this_epoch = this_line.split('\"epoch\": ')[-1].split(',')[0]
                this_epoch = int(this_epoch) #+ 1
                epochs_gan.append(this_epoch)
                this_val_acc1 = this_line.split('\"test_acc1\": ')[-1].split(',')[0]
                this_val_acc1 = float(this_val_acc1)
                valid_acc1_list_gan.append(this_val_acc1)
                
    assert(len(epochs_gan) == len(valid_acc1_list_gan))
    assert(len(epochs_gan) == 50)
    assert(epochs_orig == epochs_gan)
    
    model_type = model_arch + supCls_folder.split('/')[0]
    
    # plot curves of val_acc1_list_orig & val_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, valid_acc1_list_orig)
    ax.plot(epochs_gan, valid_acc1_list_gan)
    ax.legend(['valid_acc1_orig', 'valid_acc1_cGAN'])
    ax.set_ylim([0,max(max(valid_acc1_list_orig), max(valid_acc1_list_gan))+10])
    title_str = model_type + '_valid_acc1'
    plt.title(title_str)
    fig.savefig(dstRootDir + supCls_folder + title_str + '.png')

    # get metrics:
    orig_acc1_mean = np.mean(valid_acc1_list_orig[-5:])
    orig_acc1_max = max(valid_acc1_list_orig)
    orig_acc1_max_epoch = np.argmax(valid_acc1_list_orig)
    
    gan_acc1_mean = np.mean(valid_acc1_list_gan[-5:])
    gan_acc1_max = max(valid_acc1_list_gan)
    gan_acc1_max_epoch = np.argmax(valid_acc1_list_gan)
    
    print('np.mean(valid_acc1_list_orig[-5:]) = ' + str(orig_acc1_mean))
    print('max(valid_acc1_list_orig) = ' + str(orig_acc1_max))
    print('np.argmax(valid_acc1_list_orig) = ' + str(orig_acc1_max_epoch))
    print()
    print('np.mean(valid_acc1_list_gan[-5:]) = ' + str(gan_acc1_mean))
    print('max(valid_acc1_list_gan) = ' + str(gan_acc1_max))
    print('np.argmax(valid_acc1_list_gan) = ' + str(gan_acc1_max_epoch))


