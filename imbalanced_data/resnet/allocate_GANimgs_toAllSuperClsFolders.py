#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:42:48 2022

@author: guo.1648
"""

# referenced from code:
# /eecf/cbcsl/data100b/Chenqi/stylegan2/allocate_GANimgs_toSubClsFolders.py

import cv2
import os
import numpy as np
import shutil
import pickle


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/aug_data/iNaturalist/train/'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/aug_data/iNaturalist_merge/train/'

superCls_folder_list = ['Amphibians/', 'Birds/', 'Fungi/', 'Insects/', 'Reptiles/']


if __name__ == '__main__':
    
    # init counter dict to store statistics for the dataset:
    stat_gan_dict = {}
    
    # deal with the GAN sample imgs:
    for superCls_folder in superCls_folder_list:
        print("------------------deal with---------------------")
        print(superCls_folder)
        
        srcDir = srcRootDir + superCls_folder
        assert(os.path.exists(srcDir))
        dstDir = dstRootDir + superCls_folder
        assert(os.path.exists(dstDir))
        
        for (_, dirnames, _) in os.walk(srcDir):
            #print(dirnames)
            for this_dirname in dirnames:
                print("$$$$$$$$$$$$$$$$$$$")
                print(this_dirname)
                srcDir_subCls = srcDir + this_dirname + '/'
                assert(os.path.exists(srcDir_subCls))
                
                for (sub_dirpath, sub_dirnames, sub_filenames) in os.walk(srcDir_subCls):
                    for sub_filename in sub_filenames:
                        if (".png" in sub_filename) or (".jpg" in sub_filename):
                            #print("------------------deal with---------------------")
                            #print(filename)
                            this_img_name = srcDir_subCls + sub_filename
                            assert(os.path.exists(this_img_name))
                            
                            # copy fullImgName_GAN to dstDir_subClsImg:
                            if 'seed' in sub_filename:
                                new_img_name = dstDir + this_dirname + '_' + sub_filename
                            else:
                                new_img_name = dstDir + sub_filename
                            
                            if not (os.path.exists(new_img_name)):
                                shutil.copyfile(this_img_name, new_img_name)
            
    




