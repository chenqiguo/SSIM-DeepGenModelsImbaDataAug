#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:27:24 2022

@author: guo.1648
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/'

orig_prefix = 'cls_res18_orig_'
gan_prefix = 'cls_res18_gan_v3cls_'

folder_suff = 'iNaturalist/Fungi/'

pklFile = 'history.pkl'
txtFile = 'history_all.txt'

model_arch = 'eachSubCls_v3cls_'


if __name__ == '__main__':
    
    orig_folder = srcRootDir + orig_prefix + folder_suff
    assert(os.path.exists(orig_folder))
    gan_folder = srcRootDir + gan_prefix + folder_suff
    assert(os.path.exists(gan_folder))
    
    epochs_orig = []
    train_acc1_list_orig = []
    valid_acc1_list_orig = []
    
    epochs_gan = []
    train_acc1_list_gan = []
    valid_acc1_list_gan= []
    
    # (1) for original images:
    orig_pkl_fullName = orig_folder + pklFile
    assert(os.path.exists(orig_pkl_fullName))
    
    f_pkl = open(orig_pkl_fullName,'rb')
    history_orig = pickle.load(f_pkl)
    f_pkl.close()
    
    for dict_orig in history_orig:
        epochs_orig.append(dict_orig['epoch'])
        train_acc1_list_orig.append(dict_orig['acc1_train'].item())
        valid_acc1_list_orig.append(dict_orig['acc1_val'].item())
    
    # (2) for gan synthesized images:
    gan_txt_fullName = gan_folder + txtFile
    assert(os.path.exists(gan_txt_fullName))
    
    with open(gan_txt_fullName) as file:
        for line in file:
            this_line = line.rstrip()
            
            if '[600/607]' in this_line:
                this_epoch = this_line.split('Epoch: [')[-1].split(']')[0]
                this_epoch = int(this_epoch) + 1
                epochs_gan.append(this_epoch)
                this_train_acc1 = this_line.split('Acc@1  ')[-1].split(' ( ')[0]
                this_train_acc1 = float(this_train_acc1)
                train_acc1_list_gan.append(this_train_acc1)
            
            if '* Acc@1 ' in this_line:
                this_val_acc1 = this_line.split('* Acc@1 ')[-1].split(' Acc@5 ')[0]
                this_val_acc1 = float(this_val_acc1)
                valid_acc1_list_gan.append(this_val_acc1)
    
    assert(len(epochs_gan) == len(valid_acc1_list_gan))
    assert(len(epochs_gan) == 100)
    assert(epochs_orig == epochs_gan)
    
    model_type = model_arch + folder_suff.split('/')[0] + '_' + folder_suff.split('/')[1]
    
    # plot curves of train_acc1_list_orig & train_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, train_acc1_list_orig)
    ax.plot(epochs_gan, train_acc1_list_gan)
    ax.legend(['train_acc1_orig_res18', 'train_acc1_gan_res18'])
    ax.set_ylim([0,max(max(train_acc1_list_orig), max(train_acc1_list_gan))+10])
    title_str = model_type + '_train_acc1'
    plt.title(title_str)
    fig.savefig(srcRootDir + title_str + '.png')
    # plot curves of val_acc1_list_orig & val_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, valid_acc1_list_orig)
    ax.plot(epochs_gan, valid_acc1_list_gan)
    ax.legend(['valid_acc1_orig_res18', 'valid_acc1_gan_res18'])
    ax.set_ylim([0,max(max(valid_acc1_list_orig), max(valid_acc1_list_gan))+10])
    title_str = model_type + '_valid_acc1'
    plt.title(title_str)
    fig.savefig(srcRootDir + title_str + '.png')

    
    
    
    
    
