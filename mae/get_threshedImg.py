#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:15:15 2022

@author: guo.1648
"""

# Referenced from imgSelection_chenqi.py

# Using the all_prob_dict.pkl generated in imgSelectionMAE_chenqi.py, 
# select images with pred_prob > thresh to aug-data folder.
# Also copy orig images to aug-data folder if not did so before.
# Finally check if the num of images in each sub-class in aug-data equals the required num.


import os
import pickle
import shutil
import random

"""
srcRootDir = 'cGAN_data/generate/'
pklFileName = 'all_prob_dict.pkl'
supCls_folder = 'Birds_128-011200_tmp3/'

srcRootDir_orig = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist/train/Birds/'

prob_thresh = 3
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/aug-data/Birds/step1/thresh_' + str(prob_thresh) + '/'

total_num_needed = 1121 #993 # 2770 
"""
"""
srcRootDir = 'cGAN_data/generate/'
pklFileName = 'all_prob_dict.pkl'
supCls_folder = 'Amphibians_128-009200/'

srcRootDir_orig = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist/train/Amphibians/'

prob_thresh = 3
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/aug-data/Amphibians/step1/thresh_' + str(prob_thresh) + '/'

total_num_needed = 2770 
"""
"""
srcRootDir = 'cGAN_data/generate/'
pklFileName = 'all_prob_dict.pkl'
supCls_folder = 'Fungi_128-006800/'

srcRootDir_orig = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist/train/Fungi/'

prob_thresh = 3
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/aug-data/Fungi/step1/thresh_' + str(prob_thresh) + '/'

total_num_needed = 2770 
"""
"""
srcRootDir = 'cGAN_data/generate/'
pklFileName = 'all_prob_dict.pkl'
supCls_folder = 'flowers_128-007600/'

srcRootDir_orig = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/flowers/train/'

prob_thresh = 2
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/aug-data/flowers/step1/thresh_' + str(prob_thresh) + '/'

total_num_needed = 2770 
"""
"""
srcRootDir = 'cGAN_data/generate/'
pklFileName = 'all_prob_dict.pkl'
supCls_folder = 'UTKFace_128-004200/'

srcRootDir_orig = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/UTKFace/UTKFace/utkface_aligned_cropped/group_by_ethnicity/train/'

prob_thresh = 2
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/aug-data/UTKFace/step1/thresh_' + str(prob_thresh) + '/'

total_num_needed = 10000 
"""

srcRootDir = 'cGAN_data/generate/'
pklFileName = 'all_prob_dict.pkl'
supCls_folder = 'scene_128-004800/'

srcRootDir_orig = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene/scene/train/'

prob_thresh = 2.5
dstRootDir = '/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/aug-data/scene/step1/thresh_' + str(prob_thresh) + '/'

total_num_needed = 3000 



def get_clsFolders(this_dir):
    for (dirpath, dirnames, filenames) in os.walk(this_dir):
        return dirnames




if __name__ == '__main__':
    
    # load the prob dict pkl:
    pred_prob_dict_fName = srcRootDir + supCls_folder + pklFileName
    assert(os.path.exists(pred_prob_dict_fName))
    
    f_pkl = open(pred_prob_dict_fName,'rb')
    all_prob_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    # get sub-class folder names:
    srcDir_val = srcRootDir + supCls_folder + 'val/'
    cls_folders = get_clsFolders(srcDir_val)
    
    for this_imgFullName in all_prob_dict:
        print(this_imgFullName)
        assert(os.path.exists(this_imgFullName))
        
        this_pred_prob = all_prob_dict[this_imgFullName]
        
        this_cls_folder = this_imgFullName.split('/')[-2]
        
        this_dstDir = dstRootDir + this_cls_folder + '/'
        if not (os.path.exists(this_dstDir)):
            os.mkdir(this_dstDir)
        
        if this_pred_prob > prob_thresh: #4.5: <-- for debug!!!
            src_GAN_imgName = this_imgFullName
            dst_GAN_imgName = this_dstDir + this_imgFullName.split('/')[-1]
            #dst_GAN_imgName = dstDir_thresh + 'b-' + GAN_imgName.split('/')[-1].split('a-')[-1] # only for Amphibians b
            if os.path.exists(src_GAN_imgName) and not (os.path.exists(dst_GAN_imgName)):
                shutil.copyfile(src_GAN_imgName, dst_GAN_imgName)
        
    
    dst_cls_folders = get_clsFolders(dstRootDir)
    #assert(len(dst_cls_folders) == len(cls_folders))
    
    # (2) check if resulting GAN-syn img numbers >= the required number;
    #     if NOT, go back and generate more GAN-imgs;
    #     if so, delete the redundant GAN imgs
    for cls_folder in dst_cls_folders:
        print(cls_folder)
        this_dstDir = dstRootDir + cls_folder + '/'
        assert(os.path.exists(this_dstDir))
        GAN_file_list_ = os.listdir(this_dstDir)
        # newly added: remove those orig img names from GAN_file_list_:
        GAN_file_list = []
        for tmp_item in GAN_file_list_:
            if 'seed' in tmp_item:
                GAN_file_list.append(tmp_item)
        num_GAN_imgs = len(GAN_file_list)
        
        srcDir_orig  =  srcRootDir_orig + cls_folder + '/'
        assert(os.path.exists(srcDir_orig))
        orig_file_list = os.listdir(srcDir_orig)
        num_orig_imgs = len(orig_file_list)
        
        GAN_num_needed = total_num_needed - num_orig_imgs
        
        if (num_GAN_imgs < GAN_num_needed):
            print('***** sub-cls ' + cls_folder + ' GAN imgs num NOT enough!')
        else:
            # delete the redundant GAN imgs:
            num_GAN_toDiscard = num_GAN_imgs - GAN_num_needed
            fileList_GAN_toDiscard = random.sample(GAN_file_list, num_GAN_toDiscard)
            for fileName_GAN_toDiscard in fileList_GAN_toDiscard:
                fullFileName_GAN_toDiscard = this_dstDir + fileName_GAN_toDiscard
                if (os.path.exists(fullFileName_GAN_toDiscard)): os.remove(fullFileName_GAN_toDiscard)

    
    #"""
    # (3) copy orig imgs to dst cGAN folder:
    print('copying orig imgs...')
    for cls_folder in cls_folders:
        #print(cls_folder)
        srcDir_orig  =  srcRootDir_orig + cls_folder + '/'
        assert(os.path.exists(srcDir_orig))
        this_dstDir = dstRootDir + cls_folder + '/'
        if not (os.path.exists(this_dstDir)): os.mkdir(this_dstDir)
        
        for (dirpath, dirnames, filenames) in os.walk(srcDir_orig):
            for filename in filenames:
                if (".jpg" in filename):
                    #print("------------------deal with---------------------")
                    #print(filename)
                    src_orig_imgName = srcDir_orig + filename
                    assert(os.path.exists(src_orig_imgName))
                    dst_orig_imgName = this_dstDir + filename
                    if not (os.path.exists(dst_orig_imgName)):
                        shutil.copyfile(src_orig_imgName, dst_orig_imgName)
    #"""
    

