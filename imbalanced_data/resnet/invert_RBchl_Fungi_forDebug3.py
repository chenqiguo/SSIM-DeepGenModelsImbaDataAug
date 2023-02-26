#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:49:53 2022

@author: guo.1648
"""

# In sub-cls 3, we want red mushrooms, but they are blue in /Fungi_forDebug/
# —> HOW about doing the trick: invert B&R channels for those blue mushrooms?
# This code achieves that and save the resulting images in /Fungi_forDebug3/ .


import cv2
import numpy as np
import os


srcDstDir_img = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/aug_data_v3_cls/iNaturalist/train/Fungi_forDebug3_step3/11/'


if __name__ == '__main__':
    for (dirpath, dirnames, filenames) in os.walk(srcDstDir_img):
        for filename in filenames:
            if ("dimage1.png" in filename):
                print("------------------deal with---------------------")
                print(filename)
                img_fullName = srcDstDir_img + filename
                assert(os.path.exists(img_fullName))
                image = cv2.imread(img_fullName)
                (B, G, R) = cv2.split(image)
                image_inv = cv2.merge([R, G, B])
                cv2.imwrite(img_fullName, image_inv)

                

