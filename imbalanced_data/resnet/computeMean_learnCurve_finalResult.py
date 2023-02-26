#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:57:48 2022

@author: guo.1648
"""

# compute mean value of the last 10 epochs of the train & val learning procedure.
# this is for ECCV SM.


import os
import pickle
import numpy as np

#import matplotlib.pyplot as plt

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/'

orig_rootFolder = 'scene/cls_res18_orig/' #'cls_res18_orig_iNaturalist/'
gan_rootFolder = 'scene/cls_res18_cGAN/' #'cls_res18_cGAN_iNaturalist/' 


gan_used_step_folder_list = ['opt2/step2/thresh_30/'] #['Amphibians/opt2/step1/thresh_10/', 'Birds/opt2/step1/thresh_10/', 'Fungi/opt2/step2/based on step1 thresh10/thresh_10/', 'Insects/opt2/step2/thresh_10/', 'Reptiles/opt2/step2/thresh_20/']

pklFile = 'history.pkl'

model_arch = 'scene/' #'iNaturalist_cGANaug_'


"""
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
"""


if __name__ == '__main__':
    
    for idx, gan_used_step_folder in enumerate(gan_used_step_folder_list):
        
        print('***************** for iNaturalist ' + model_arch.split('/')[0] + ':')
        
        # (1) for original images:
        folder_suff_orig = model_arch
        orig_folder = srcRootDir + orig_rootFolder #+ folder_suff_orig
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
        gan_folder = srcRootDir + gan_rootFolder + gan_used_step_folder
        assert(os.path.exists(gan_folder))
        
        epochs_gan = []
        train_acc1_list_gan = []
        valid_acc1_list_gan= []
        
        gan_pkl_fullName = gan_folder + pklFile
        assert(os.path.exists(gan_pkl_fullName))
        
        f_pkl = open(gan_pkl_fullName,'rb')
        history_gan = pickle.load(f_pkl)
        f_pkl.close()
        
        for dict_gan in history_gan:
            epochs_gan.append(dict_gan['epoch'])
            train_acc1_list_gan.append(dict_gan['acc1_train'].item())
            valid_acc1_list_gan.append(dict_gan['acc1_val'].item())
        
        
        # print the results:
        valid_acc1_orig_mean = np.mean(valid_acc1_list_orig[-5:])
        valid_acc1_gan_mean = np.mean(valid_acc1_list_gan[-5:])
        
        print('np.mean(valid_acc1_list_orig[-5:]) = ' + str(valid_acc1_orig_mean))
        print('np.mean(valid_acc1_list_gan[-5:]) = ' + str(valid_acc1_gan_mean))
    
    
    
    
    
    


