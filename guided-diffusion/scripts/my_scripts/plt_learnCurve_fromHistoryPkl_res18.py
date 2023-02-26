#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:50:12 2021

@author: guo.1648
"""


import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/'

"""
### for Fungi:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Fungi/'
gan_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step2/based on step1 thresh10/thresh_10/'

pklFile = 'history.pkl'
model_arch = 'iNaturalist_Fungi'

# newly added for plotting diffusion model result curve:
srcDstDir_diffu = '/eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/'

diffu_prefix = 'Fungi_new/step1/thresh_10/'
"""
"""
### for Birds:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Birds/'
gan_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step1/thresh_10/'

pklFile = 'history.pkl'

model_arch = 'iNaturalist_Birds'


# newly added for plotting diffusion model result curve:
srcDstDir_diffu = '/eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/'

diffu_prefix = 'Birds/step1/thresh_10/'
"""
"""
### for scene:
orig_prefix = 'scene/cls_res18_orig/'
gan_prefix = 'scene/cls_res18_cGAN/opt2/step2/thresh_30/'

pklFile = 'history.pkl'

model_arch = 'scene'


# newly added for plotting diffusion model result curve:
srcDstDir_diffu = '/eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/'

diffu_prefix = 'scene/step1/thresh_10/'
"""

### for Amphibians:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Amphibians/'
gan_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Amphibians/opt2/step1/thresh_10/'

pklFile = 'history.pkl'

model_arch = 'iNaturalist_Amphibians'


# newly added for plotting diffusion model result curve:
srcDstDir_diffu = '/eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/'

diffu_prefix = 'Amphibians/step1/thresh_10/'




if __name__ == '__main__':
    
    orig_folder = srcRootDir + orig_prefix #+ folder_suff
    assert(os.path.exists(orig_folder))
    gan_folder = srcRootDir + gan_prefix #+ folder_suff
    assert(os.path.exists(gan_folder))
    diffu_folder = srcDstDir_diffu + diffu_prefix
    assert(os.path.exists(diffu_folder))
    
    epochs_orig = []
    train_acc1_list_orig = []
    valid_acc1_list_orig = []
    
    epochs_gan = []
    train_acc1_list_gan = []
    valid_acc1_list_gan= []
    
    epochs_diffu = []
    train_acc1_list_diffu = []
    valid_acc1_list_diffu= []
    
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
    gan_pkl_fullName = gan_folder + pklFile
    assert(os.path.exists(gan_pkl_fullName))
    
    f_pkl = open(gan_pkl_fullName,'rb')
    history_gan = pickle.load(f_pkl)
    f_pkl.close()
    
    for dict_gan in history_gan:
        epochs_gan.append(dict_gan['epoch'])
        train_acc1_list_gan.append(dict_gan['acc1_train'].item())
        valid_acc1_list_gan.append(dict_gan['acc1_val'].item())
    
    # (3) for diffu synthesized images:
    diffu_pkl_fullName = diffu_folder + pklFile
    assert(os.path.exists(diffu_pkl_fullName))

    f_pkl = open(diffu_pkl_fullName,'rb')
    history_diffu = pickle.load(f_pkl)
    f_pkl.close()

    for dict_diffu in history_diffu:
        epochs_diffu.append(dict_diffu['epoch'])
        train_acc1_list_diffu.append(dict_diffu['acc1_train'].item())
        valid_acc1_list_diffu.append(dict_diffu['acc1_val'].item())
    
    
    assert(epochs_orig == epochs_gan)
    assert(epochs_orig == epochs_diffu)
    
    model_type = model_arch #+ folder_suff.split('/')[0] + '_' + folder_suff.split('/')[1]
    
    # plot curves of train_acc1_list_orig & train_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, train_acc1_list_orig)
    ax.plot(epochs_gan, train_acc1_list_gan)
    ax.plot(epochs_diffu, train_acc1_list_diffu)
    ax.legend(['train_acc1_orig_res18', 'train_acc1_gan_res18', 'train_acc1_diffusion_res18'])
    ax.set_ylim([0,max(max(train_acc1_list_orig), max(train_acc1_list_gan))+10])
    title_str = model_type + '_train_acc1'
    plt.title(title_str)
    fig.savefig(diffu_folder + title_str + '.png')
    # plot curves of val_acc1_list_orig & val_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, valid_acc1_list_orig)
    ax.plot(epochs_gan, valid_acc1_list_gan)
    ax.plot(epochs_diffu, valid_acc1_list_diffu)
    ax.legend(['valid_acc1_orig_res18', 'valid_acc1_gan_res18', 'valid_acc1_diffusion_res18'])
    ax.set_ylim([0,max(max(valid_acc1_list_orig), max(valid_acc1_list_gan))+10])
    title_str = model_type + '_valid_acc1'
    plt.title(title_str)
    fig.savefig(diffu_folder + title_str + '.png')
    
    print('max(valid_acc1_list_orig)='+str(max(valid_acc1_list_orig)))
    print('max(valid_acc1_list_gan)='+str(max(valid_acc1_list_gan)))
    print('at Epoch: ' + str(np.argmax(valid_acc1_list_gan)))
    print('max(valid_acc1_list_diffu)='+str(max(valid_acc1_list_diffu)))
    print('at Epoch: ' + str(np.argmax(valid_acc1_list_diffu)))
    
    print('np.mean(valid_acc1_list_orig[-5:]) = ' + str(np.mean(valid_acc1_list_orig[-5:])))
    print('np.mean(valid_acc1_list_gan[-5:]) = ' + str(np.mean(valid_acc1_list_gan[-5:])))
    print('np.mean(valid_acc1_list_diffu[-5:]) = ' + str(np.mean(valid_acc1_list_diffu[-5:])))





