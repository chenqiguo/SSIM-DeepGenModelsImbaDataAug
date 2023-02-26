#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:09:31 2022

@author: guo.1648
"""

# rename the images for scene dataset: add class label + underscore for each image file name!

import os
import shutil
import sys


rootDir_img = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene/scene/'

folders = ['train/', 'val/']

subClses = ['0/', '1/', '2/', '3/', '4/', '5/']



if __name__ == '__main__':
    
    for folder in folders:
        dir_img = rootDir_img + folder
        
        for subCls in subClses:
            dir_img_thisCls = dir_img + subCls
            print("------------------deal with---------------------")
            print(dir_img_thisCls)
            
            for (dirpath, dirnames, filenames) in os.walk(dir_img_thisCls):
                for filename in filenames:
                    if ".jpg" in filename:
                        #print(filename)
                        fullFname = dir_img_thisCls + filename
                        assert(os.path.exists(fullFname))
                        
                        new_fullFname = dir_img_thisCls + subCls.split('/')[0] + '_' + filename
                        
                        os.rename(fullFname, new_fullFname)
                        #print()
                            


