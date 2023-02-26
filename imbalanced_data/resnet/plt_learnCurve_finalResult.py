#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:02:20 2022

@author: guo.1648
"""

# this code is to get the final learning curve plot as result in the paper, by putting them together.

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/'

orig_prefix = 'cls_res18_orig_'
gan_prefix = 'cls_res18_gan_v3'

folder_suff_orig = 'iNaturalist/Fungi/'
folder_suff_gan = 'cls_iNaturalist/Fungi/'

gan_used_step_folder_list = ['forDebug/', 'forDebug_step2/', 'forDebug_step3_v2/thresh_40/',
                             'forDebug_step4_v2/thresh_20/']

pklFile = 'history.pkl'

model_arch = 'eachSubCls_v3cls_'




def get_history_func(gan_folder, flag):
    epochs_gan = []
    train_acc1_list_gan = []
    valid_acc1_list_gan= []
    
    if flag == 'part':
        for i in range(1,6):
            #print(i)
            partF = 'part' + str(i) + '/'
            
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
    
    else: #flag == 'all'
        gan_pkl_fullName = gan_folder + pklFile
        assert(os.path.exists(gan_pkl_fullName))
        
        f_pkl = open(gan_pkl_fullName,'rb')
        history_gan = pickle.load(f_pkl)
        f_pkl.close()
        
        for dict_gan in history_gan:
            epochs_gan.append(dict_gan['epoch'])
            train_acc1_list_gan.append(dict_gan['acc1_train'].item())
            valid_acc1_list_gan.append(dict_gan['acc1_val'].item())
    
    
    return (epochs_gan, train_acc1_list_gan, valid_acc1_list_gan)




if __name__ == '__main__':
    
    # (1) for original images:
    orig_folder = srcRootDir + orig_prefix + folder_suff_orig
    assert(os.path.exists(orig_folder))
    
    epochs_orig = []
    train_acc1_list_orig = []
    valid_acc1_list_orig = []
    
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
    epochs_gan_dict = {}
    train_acc1_list_gan_dict = {}
    valid_acc1_list_gan_dict = {}
    
    for idx, gan_used_step_folder in enumerate(gan_used_step_folder_list):
        gan_folder = srcRootDir + gan_prefix + folder_suff_gan + gan_used_step_folder
        assert(os.path.exists(gan_folder))
        
        if gan_used_step_folder == 'forDebug/':
            epochs_gan, train_acc1_list_gan, valid_acc1_list_gan = get_history_func(gan_folder, flag='part')
        else:
            epochs_gan, train_acc1_list_gan, valid_acc1_list_gan = get_history_func(gan_folder, flag='all')
        
        epochs_gan_dict[idx] = epochs_gan
        train_acc1_list_gan_dict[idx] = train_acc1_list_gan
        valid_acc1_list_gan_dict[idx] = valid_acc1_list_gan

    # (3) plot all the learning curves together as final result:
    assert(epochs_orig == epochs_gan)
    
    # plot curves of train_acc1_list_orig & train_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, train_acc1_list_orig)
    ax.plot(epochs_gan_dict[0], train_acc1_list_gan_dict[0])
    ax.plot(epochs_gan_dict[1], train_acc1_list_gan_dict[1])
    ax.plot(epochs_gan_dict[2], train_acc1_list_gan_dict[2])
    ax.plot(epochs_gan_dict[3], train_acc1_list_gan_dict[3])
    ax.legend(['orig', 'GLMNS-cls step1', 'GLMNS-cls step2', 'GLMNS-cls step3', 'GLMNS-cls step4'])
    ax.set_ylim([0,max(max(train_acc1_list_orig), max(train_acc1_list_gan_dict[0]), max(train_acc1_list_gan_dict[1]), max(train_acc1_list_gan_dict[2]), max(train_acc1_list_gan_dict[3]))+10])
    title_str = 'iNaturalist-2019 Fungi train acc1'
    plt.title(title_str)
    fig.savefig(srcRootDir + title_str + '.png')
    # plot curves of val_acc1_list_orig & val_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, valid_acc1_list_orig)
    ax.plot(epochs_gan_dict[0], valid_acc1_list_gan_dict[0])
    ax.plot(epochs_gan_dict[1], valid_acc1_list_gan_dict[1])
    ax.plot(epochs_gan_dict[2], valid_acc1_list_gan_dict[2])
    ax.plot(epochs_gan_dict[3], valid_acc1_list_gan_dict[3])
    ax.legend(['orig', 'GLMNS-cls step1', 'GLMNS-cls step2', 'GLMNS-cls step3', 'GLMNS-cls step4'])
    ax.set_ylim([0,max(max(valid_acc1_list_orig), max(valid_acc1_list_gan_dict[0]), max(valid_acc1_list_gan_dict[1]), max(valid_acc1_list_gan_dict[2]), max(valid_acc1_list_gan_dict[3]))+10])
    title_str = 'iNaturalist-2019 Fungi valid acc1'
    plt.title(title_str)
    fig.savefig(srcRootDir + title_str + '.png')
    
    
    
    # newly added: to compute the mean of the last 10 epochs' acc as the value reported in our paper:
    train_acc1_orig_mean = np.mean(train_acc1_list_orig[-10:])
    train_acc1_gan_0_mean= np.mean(train_acc1_list_gan_dict[0][-10:])
    train_acc1_gan_1_mean= np.mean(train_acc1_list_gan_dict[1][-10:])
    train_acc1_gan_2_mean= np.mean(train_acc1_list_gan_dict[2][-10:])
    train_acc1_gan_3_mean= np.mean(train_acc1_list_gan_dict[3][-10:])
    
    valid_acc1_orig_mean = np.mean(valid_acc1_list_orig[-10:])
    valid_acc1_gan_0_mean= np.mean(valid_acc1_list_gan_dict[0][-10:])
    valid_acc1_gan_1_mean= np.mean(valid_acc1_list_gan_dict[1][-10:])
    valid_acc1_gan_2_mean= np.mean(valid_acc1_list_gan_dict[2][-10:])
    valid_acc1_gan_3_mean= np.mean(valid_acc1_list_gan_dict[3][-10:])
    
    print('train_acc1_orig_mean = ' + str(train_acc1_orig_mean))
    print('train_acc1_gan_0_mean = ' + str(train_acc1_gan_0_mean))
    print('train_acc1_gan_1_mean = ' + str(train_acc1_gan_1_mean))
    print('train_acc1_gan_2_mean = ' + str(train_acc1_gan_2_mean))
    print('train_acc1_gan_3_mean = ' + str(train_acc1_gan_3_mean))
    
    print('valid_acc1_orig_mean = ' + str(valid_acc1_orig_mean))
    print('valid_acc1_gan_0_mean = ' + str(valid_acc1_gan_0_mean))
    print('valid_acc1_gan_1_mean = ' + str(valid_acc1_gan_1_mean))
    print('valid_acc1_gan_2_mean = ' + str(valid_acc1_gan_2_mean))
    print('valid_acc1_gan_3_mean = ' + str(valid_acc1_gan_3_mean))
    
    
    
