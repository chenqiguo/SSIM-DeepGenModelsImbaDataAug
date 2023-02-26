#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:06:51 2022

@author: guo.1648
"""

# for iNaturalist_allSubCls experiments!!!

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/'

orig_prefix = 'cls_res18_orig_'
gan_prefix = 'cls_res18_gan_'

folder_suff = 'iNaturalist/'

pklFile = 'history.pkl'

model_arch = 'allSubCls_'


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
    
    #"""
    # (1) for original images:
    for i in range(1,6):
        #print(i)
        partF = 'part' + str(i) + '/'
        orig_pkl_fullName = orig_folder + partF + pklFile
        assert(os.path.exists(orig_pkl_fullName))
        
        f_pkl = open(orig_pkl_fullName,'rb')
        history_orig = pickle.load(f_pkl)
        f_pkl.close()
        
        if i != 1:
            # for orig:
            orig_starting_epoch = history_orig[0]['epoch'] - 1
            epochs_orig = epochs_orig[:orig_starting_epoch]
            train_acc1_list_orig = train_acc1_list_orig[:orig_starting_epoch]
            valid_acc1_list_orig = valid_acc1_list_orig[:orig_starting_epoch]
        
        for dict_orig in history_orig:
            epochs_orig.append(dict_orig['epoch'])
            train_acc1_list_orig.append(dict_orig['acc1_train'].item())
            valid_acc1_list_orig.append(dict_orig['acc1_val'].item())
    #"""
    
    # (2) for gan synthesized images:
    for i in range(1,7):
        #print(i)
        partF = 'part' + str(i) + '/'
        
        if i != 3: # part3 is using history.txt file!
            gan_pkl_fullName = gan_folder + partF + pklFile
            assert(os.path.exists(gan_pkl_fullName))
            
            f_pkl = open(gan_pkl_fullName,'rb')
            history_gan = pickle.load(f_pkl)
            f_pkl.close()
            
            if i != 1:
                # for gan:
                gan_starting_epoch = history_gan[0]['epoch'] - 1
                epochs_gan = epochs_gan[:gan_starting_epoch]
                train_acc1_list_gan = train_acc1_list_gan[:gan_starting_epoch]
                valid_acc1_list_gan = valid_acc1_list_gan[:gan_starting_epoch]
            
            for dict_gan in history_gan:
                epochs_gan.append(dict_gan['epoch'])
                train_acc1_list_gan.append(dict_gan['acc1_train'].item())
                valid_acc1_list_gan.append(dict_gan['acc1_val'].item())
        
        # for part3 gan folder:
        else: # i==3
            #print('&&&&&&&&&&&&& HERE 1 !!!!')
            gan_txt_fullName = gan_folder + partF + 'hist_output.txt'
            assert(os.path.exists(gan_txt_fullName))
            
            # part3 starts from epoch 32:
            epochs_gan = epochs_gan[:32]
            train_acc1_list_gan = train_acc1_list_gan[:32]
            valid_acc1_list_gan = valid_acc1_list_gan[:32]
            
            with open(gan_txt_fullName) as file:
                for line in file:
                    this_line = line.rstrip()
                    
                    if '[7060/7062]' in this_line:
                        this_epoch = this_line.split('Epoch: [')[-1].split(']')[0]
                        this_epoch = int(this_epoch) + 1
                        epochs_gan.append(this_epoch)
                        this_train_acc1 = this_line.split('Acc@1  ')[-1].split(' ( 3')[0]
                        this_train_acc1 = float(this_train_acc1)
                        train_acc1_list_gan.append(this_train_acc1)
                    
                    if '* Acc@1 ' in this_line:
                        this_val_acc1 = this_line.split('* Acc@1 ')[-1].split(' Acc@5 ')[0]
                        this_val_acc1 = float(this_val_acc1)
                        valid_acc1_list_gan.append(this_val_acc1)
    
    
    model_type = model_arch + folder_suff.split('/')[0]
    
    #"""
    # only use first 50 epochs:
    epochs_orig = epochs_orig[:50]
    epochs_gan = epochs_gan[:50]
    train_acc1_list_orig = train_acc1_list_orig[:50]
    train_acc1_list_gan = train_acc1_list_gan[:50]
    valid_acc1_list_orig = valid_acc1_list_orig[:50]
    valid_acc1_list_gan = valid_acc1_list_gan[:50]
    #"""
    
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

    
    
    
    





