#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:57:48 2022

@author: guo.1648
"""

# using the results.pkl file generated from cls_res_test.py,
# allocate the classified images to corresponding subclass folders.

import cv2
import os
import numpy as np
import shutil
import pickle


# NOTE: This code is just used to generate pred_mapto_actual_dict_v2 for GLMNS-cls.
# Thus these images are using fake-cls-idx (i.e., each sub-cls starting from class 0 --> results_tmp.pkl)
# and ONLY contains original images (i.e., NO GAN-syn images here)!!!

srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/aug_data_v3/iNaturalist/train/Fungi/'

# Note: What we need is results_tmp.pkl (instead of results.pkl)!!!:
srcPkl_fullName = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi/results_tmp.pkl'

dstRootDir_img = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/aug_data_v3_cls/iNaturalist_post/train/Fungi/'


if __name__ == '__main__':
    
    # load the pkl file:
    f_pkl = open(srcPkl_fullName, 'rb')
    results = pickle.load(f_pkl)
    f_pkl.close()
    
    # init counter dict to store statistics for the dataset:
    stat_gan_dict = {}
    
    for imgOrigName in results:
        
        if 'dimage1' in imgOrigName: # ONLY contains original images (i.e., NO GAN-syn images here)!!!
            continue
        
        actual_cls = str(results[imgOrigName])
        
        if not (os.path.exists(dstRootDir_img+actual_cls)):
            os.makedirs(dstRootDir_img+actual_cls)
        
        srcImg_fullName = srcRootDir_img + imgOrigName
        assert(os.path.exists(srcImg_fullName))
        dstImg_fullName = dstRootDir_img+actual_cls + '/' + imgOrigName.split('/')[-1]
        
        if not (os.path.exists(dstImg_fullName)):
            shutil.copyfile(srcImg_fullName, dstImg_fullName)
            
            if actual_cls not in stat_gan_dict:
                stat_gan_dict[actual_cls] = 1
            else:
                stat_gan_dict[actual_cls] += 1

    print('stat_gan_dict = ' + str(stat_gan_dict))




