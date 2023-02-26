#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 12:04:00 2022

@author: guo.1648
"""


# referenced from plt_learnCurve_fromHistoryPkl_res18.py

# according to reviewer2: adding the classification results of cGAN-aug data
# WITHOUT cls-select as baseline!


import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/'

"""
## for Amphibians:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Amphibians/'
ganCls_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Amphibians/opt2/step1/thresh_10/'
ganBase_prefix = 'cGANaug_without_cls_select/Amphibians/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_Amphibians'
"""
"""
## for Birds:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Birds/'
ganCls_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step1/thresh_10/'
ganBase_prefix = 'cGANaug_without_cls_select/Birds/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_Birds'
"""
"""
## for Fungi:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Fungi/'
ganCls_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step2/based on step1 thresh10/thresh_10/'
ganBase_prefix = 'cGANaug_without_cls_select/Fungi/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_Fungi'
"""
"""
## for Insects:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Insects/'
ganCls_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Insects/opt2/step2/thresh_10/'
ganBase_prefix = 'cGANaug_without_cls_select/Insects/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_Insects'
"""
"""
## for Reptiles:
orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Reptiles/'
ganCls_prefix = 'iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Reptiles/opt2/step2/thresh_20/'
ganBase_prefix = 'cGANaug_without_cls_select/Reptiles/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_Reptiles'
"""
"""
## for flowers:
orig_prefix = 'flowers/cls_res18_orig/'
ganCls_prefix = 'flowers/cls_res18_cGAN/opt2/step2/thresh_20/'
ganBase_prefix = 'cGANaug_without_cls_select/flowers/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_flowers'
"""
"""
## for scene:
orig_prefix = 'scene/cls_res18_orig/'
ganCls_prefix = 'scene/cls_res18_cGAN/opt2/step2/thresh_30/'
ganBase_prefix = 'cGANaug_without_cls_select/scene/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_scene'
"""

## for UTKFace:
orig_prefix = 'UTKFace/cls_res18_orig/'
ganCls_prefix = 'UTKFace/cls_res18_cGAN/opt2/step2/thresh_25/'
ganBase_prefix = 'cGANaug_without_cls_select/UTKFace/' # the results of without_cls_select

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_UTKFace'



dstDir = srcRootDir + ganBase_prefix




if __name__ == '__main__':
    
    orig_folder = srcRootDir + orig_prefix
    assert(os.path.exists(orig_folder))
    ganCls_folder = srcRootDir + ganCls_prefix
    assert(os.path.exists(ganCls_folder))
    ganBase_folder = srcRootDir + ganBase_prefix
    assert(os.path.exists(ganBase_folder))
    
    epochs_orig = []
    train_acc1_list_orig = []
    valid_acc1_list_orig = []
    
    epochs_ganCls = []
    train_acc1_list_ganCls = []
    valid_acc1_list_ganCls= []
    
    epochs_ganBase = []
    train_acc1_list_ganBase = []
    valid_acc1_list_ganBase= []
    
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
    
    # (2) for ganCls synthesized images:
    ganCls_pkl_fullName = ganCls_folder + pklFile
    assert(os.path.exists(ganCls_pkl_fullName))
    
    f_pkl = open(ganCls_pkl_fullName,'rb')
    history_ganCls = pickle.load(f_pkl)
    f_pkl.close()
    
    for dict_ganCls in history_ganCls:
        epochs_ganCls.append(dict_ganCls['epoch'])
        train_acc1_list_ganCls.append(dict_ganCls['acc1_train'].item())
        valid_acc1_list_ganCls.append(dict_ganCls['acc1_val'].item())
    
    # (3) for ganBase synthesized images:
    ganBase_pkl_fullName = ganBase_folder + pklFile
    assert(os.path.exists(ganBase_pkl_fullName))
    
    f_pkl = open(ganBase_pkl_fullName,'rb')
    history_ganBase = pickle.load(f_pkl)
    f_pkl.close()
    
    for dict_ganBase in history_ganBase:
        epochs_ganBase.append(dict_ganBase['epoch'])
        train_acc1_list_ganBase.append(dict_ganBase['acc1_train'].item())
        valid_acc1_list_ganBase.append(dict_ganBase['acc1_val'].item())
    
    
    assert(epochs_orig == epochs_ganCls)
    assert(epochs_orig == epochs_ganBase)
    
    model_type = model_arch 
    
    # plot curves of train_acc1_list_orig & train_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, train_acc1_list_orig)
    ax.plot(epochs_ganCls, train_acc1_list_ganCls)
    ax.plot(epochs_ganBase, train_acc1_list_ganBase)
    ax.legend(['train_acc1_orig_res18', 'train_acc1_ganCls_res18', 'train_acc1_ganBase_res18'])
    ax.set_ylim([0,max(max(train_acc1_list_orig), max(train_acc1_list_ganCls), max(train_acc1_list_ganBase))+10])
    title_str = model_type + '_train_acc1'
    plt.title(title_str)
    fig.savefig(dstDir + title_str + '.png')
    # plot curves of val_acc1_list_orig & val_acc1_list_gan:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs_orig, valid_acc1_list_orig)
    ax.plot(epochs_ganCls, valid_acc1_list_ganCls)
    ax.plot(epochs_ganBase, valid_acc1_list_ganBase)
    ax.legend(['valid_acc1_orig_res18', 'valid_acc1_ganCls_res18', 'valid_acc1_ganBase_res18'])
    ax.set_ylim([0,max(max(valid_acc1_list_orig), max(valid_acc1_list_ganCls), max(valid_acc1_list_ganBase))+10])
    title_str = model_type + '_valid_acc1'
    plt.title(title_str)
    fig.savefig(dstDir + title_str + '.png')
    
    print('***max(valid_acc1_list_orig)='+str(max(valid_acc1_list_orig)))
    print('***max(valid_acc1_list_ganCls)='+str(max(valid_acc1_list_ganCls)))
    print('at Epoch: ' + str(np.argmax(valid_acc1_list_ganCls)))
    print('***max(valid_acc1_list_ganBase)='+str(max(valid_acc1_list_ganBase)))
    print('at Epoch: ' + str(np.argmax(valid_acc1_list_ganBase)))
    print()
    print('***np.mean(valid_acc1_list_orig[-5:]) = ' + str(np.mean(valid_acc1_list_orig[-5:])))
    print('***np.mean(valid_acc1_list_ganCls[-5:]) = ' + str(np.mean(valid_acc1_list_ganCls[-5:])))
    print('***np.mean(valid_acc1_list_ganBase[-5:]) = ' + str(np.mean(valid_acc1_list_ganBase[-5:])))
    



