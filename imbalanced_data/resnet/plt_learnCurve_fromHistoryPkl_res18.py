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

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/'

orig_prefix = 'iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Amphibians/' #'iNaturalist_allButPlants/' #'cls_res18_orig/' #'cls_res18_orig_'
gan_prefix = 'cGANaug_without_cls_select/Amphibians_2ndTry/' #'iNaturalist_cGANaug_allButPlants/' #'cls_res18_cGAN/opt2/step2/thresh_30/' #'cls_res18_gan_v3_'

folder_suff = '' #'iNaturalist/Fungi/'

pklFile = 'history.pkl'

model_arch = 'cGAN_iNaturalist_Amphibians' #'cGAN_iNaturalist' #'cGAN_scene' #'eachSubCls_v3_'




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
    gan_pkl_fullName = gan_folder + pklFile
    assert(os.path.exists(gan_pkl_fullName))
    
    f_pkl = open(gan_pkl_fullName,'rb')
    history_gan = pickle.load(f_pkl)
    f_pkl.close()
    
    for dict_gan in history_gan:
        epochs_gan.append(dict_gan['epoch'])
        train_acc1_list_gan.append(dict_gan['acc1_train'].item())
        valid_acc1_list_gan.append(dict_gan['acc1_val'].item())
    
    assert(epochs_orig == epochs_gan)
    
    model_type = model_arch #+ folder_suff.split('/')[0] + '_' + folder_suff.split('/')[1]
    
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
    
    print('max(valid_acc1_list_orig)='+str(max(valid_acc1_list_orig)))
    print('max(valid_acc1_list_gan)='+str(max(valid_acc1_list_gan)))
    print('at Epoch: ' + str(np.argmax(valid_acc1_list_gan)))
    
    print('np.mean(valid_acc1_list_orig[-5:]) = ' + str(np.mean(valid_acc1_list_orig[-5:])))
    print('np.mean(valid_acc1_list_gan[-5:]) = ' + str(np.mean(valid_acc1_list_gan[-5:])))
    





