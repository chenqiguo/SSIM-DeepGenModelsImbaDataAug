#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:05:35 2022

@author: guo.1648
"""

# after getting the results.pkl from code find_orgCls_chenqi_step1.py,
# combining with the corresponding dataset.json file,
# get the final cGAN_cls -> orig_cls map!!!


import os
import pickle
import json


rootDir = '/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/scene_128_map/' #'/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/flowers_128_map/' #'/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Amphibians_128_map/'
srcRootDir_img = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene/scene/train/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/flowers/train/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist/train/Amphibians/'

jsn_fileName = rootDir + 'dataset.json'

pkl_fileName = rootDir + 'results.pkl'


def get_key_from_item(item, dict_):
    keys = []
    for key_tmp in dict_:
        if dict_[key_tmp] == item:
            keys.append(key_tmp)
    
    return keys


def get_clsFolders(this_dir):
    for (dirpath, dirnames, filenames) in os.walk(this_dir):
        return dirnames


def get_orig_cls_counter_dict(srcRootDir_img):
    # get img nums within each orig_cls
    
    cls_folders = get_clsFolders(srcRootDir_img)
    orig_cls_counter_dict = {} # keys are orig_cls, items are img nums
    for cls_folder in cls_folders:
        assert(os.path.exists(srcRootDir_img+cls_folder))
        number_files = len(os.listdir(srcRootDir_img+cls_folder))
        orig_cls_counter_dict[int(cls_folder)] = number_files
    
    return orig_cls_counter_dict


if __name__ == '__main__':
    f_jsn = open(jsn_fileName)
    dataset = json.load(f_jsn)
    f_jsn.close()
    
    f_pkl = open(pkl_fileName,'rb')
    results = pickle.load(f_pkl)
    f_pkl.close()
    
    assert(len(dataset['labels']) == len(results))
    
    # get img nums within each orig_cls:
    orig_cls_counter_dict = get_orig_cls_counter_dict(srcRootDir_img) # keys are orig_cls, items are img nums
    # use img numbers for a cGAN_cls to find its corresponding orig_cls:
    cGAN_cls_counter_dict = {} # count image numbers within each cGAN_cls: keys are cGAN_cls, items are img nums
    for this_item in dataset['labels']:
        # get the cGAN_cls:
        cGAN_cls = this_item[1]
        if cGAN_cls not in cGAN_cls_counter_dict:
            cGAN_cls_counter_dict[cGAN_cls] = 1
        else:
            cGAN_cls_counter_dict[cGAN_cls] += 1
    # find:
    # keys are int cGAN_cls, and items are list of corresponding all potential int orig_cls (that has same img num as cGAN_cls).
    potential_cGAN_to_orig_cls_map = {}
    for cGAN_cls in cGAN_cls_counter_dict:
        this_cGAN_cls_cnt = cGAN_cls_counter_dict[cGAN_cls]
        potential_orig_cls_list = get_key_from_item(this_cGAN_cls_cnt, orig_cls_counter_dict)
        potential_cGAN_to_orig_cls_map[cGAN_cls] = potential_orig_cls_list
    
    # the final cGAN_cls -> orig_cls map:
    # keys are int cGAN_cls, and items are list of corresponding all potential int orig_cls.
    cGAN_to_orig_cls_map_tmp = {} # get this also based on potential_cGAN_to_orig_cls_map!!!
    
    for this_item in dataset['labels']:
        imgcGANName = this_item[0]
        # get the cGAN_cls:
        cGAN_cls = this_item[1]
        # get the orig_cls:
        orig_cls = int(results[imgcGANName]) # this is based on classifier pred!!!
        if orig_cls in potential_cGAN_to_orig_cls_map[cGAN_cls]: # this is based on class image numbers!!!
            if cGAN_cls not in cGAN_to_orig_cls_map_tmp:
                cGAN_to_orig_cls_map_tmp[cGAN_cls] = [orig_cls]
            else:
                cGAN_to_orig_cls_map_tmp[cGAN_cls].append(orig_cls)
    
    if len(cGAN_to_orig_cls_map_tmp) != len(potential_cGAN_to_orig_cls_map):
        for k in potential_cGAN_to_orig_cls_map:
            if k not in cGAN_to_orig_cls_map_tmp:
                print('@@@@@@@@@@@')
                print('cGAN_cls ' + str(k) + ' missing in cGAN_to_orig_cls_map_tmp')
    
    # from cGAN_to_orig_cls_map_tmp, use the majority of orig_cls for each cGAN_cls
    # as the final cGAN_cls:
    cGAN_to_orig_cls_map_final = {}
    cGAN_to_orig_cls_map_tmp2 = {}
    orig_cls_maj_list = [] # just for below checking!
    for cGAN_cls in cGAN_to_orig_cls_map_tmp:
        List = cGAN_to_orig_cls_map_tmp[cGAN_cls]
        orig_cls_maj = max(set(List), key = List.count)
        cGAN_to_orig_cls_map_final[cGAN_cls] = orig_cls_maj
        cGAN_to_orig_cls_map_tmp2[cGAN_cls] = set(List)
        orig_cls_maj_list.append(orig_cls_maj)
    
    # check if all these orig_cls_maj's are different and contain all the original class labels:
    orig_cls_maj_set = set(orig_cls_maj_list)
    
    if len(orig_cls_maj_set) != len(cGAN_to_orig_cls_map_final):
        print('There are cGAN_cls\'es that are predicted as same orig_cls_maj!')
        dup_orig_cls_maj = [x for x in orig_cls_maj_list if orig_cls_maj_list.count(x) > 1]
        not_to_touch_cGAN_cls = set([y for y in cGAN_to_orig_cls_map_final if y not in set(dup_orig_cls_maj)])
        for dup_itm in set(dup_orig_cls_maj):
            
            cGAN_clses = get_key_from_item(dup_itm, cGAN_to_orig_cls_map_final)
            
            for dup_cGAN_cls in cGAN_clses:
                print('**********')
                print('cGAN_to_orig_cls_map_final['+str(dup_cGAN_cls)+']='+str(cGAN_to_orig_cls_map_final[dup_cGAN_cls]))
                
                for not_to_touch in not_to_touch_cGAN_cls:
                    potential_to_touch_cGAN_cls = cGAN_to_orig_cls_map_tmp2[dup_cGAN_cls]
                    if not_to_touch in potential_to_touch_cGAN_cls:
                        potential_to_touch_cGAN_cls.remove(not_to_touch)
                
                print('cGAN_to_orig_cls_map_tmp2['+str(dup_cGAN_cls)+']='+str(potential_to_touch_cGAN_cls))
            
        # modify each time:
        # here, inspect into images, and then manually run to modify our cGAN_to_orig_cls_map_final:
        # cGAN_to_orig_cls_map_final[xxx] = ...
        print()
        ### for Reptiles:
        cGAN_to_orig_cls_map_final[20]=173
        cGAN_to_orig_cls_map_final[35]=180
        cGAN_to_orig_cls_map_final[4]=165
        cGAN_to_orig_cls_map_final[5]=167
        ### for Amphibians:
        cGAN_to_orig_cls_map_final[4]=160
        cGAN_to_orig_cls_map_final[9]=153
        cGAN_to_orig_cls_map_final[0]=161
        cGAN_to_orig_cls_map_final[6]=155
        cGAN_to_orig_cls_map_final[2]=156
        cGAN_to_orig_cls_map_final[3]=159
        
    
    cls_folders = get_clsFolders(srcRootDir_img)
    if len(cGAN_to_orig_cls_map_final) != len(cls_folders):
        orig_cls_fianl_list = list(cGAN_to_orig_cls_map_final.values())
        for k_s in cls_folders:
            k = int(k_s)
            if k not in orig_cls_fianl_list:
                print('^^^^^^^^^^^^')
                print('orig_cls ' + str(k_s) + ' missing in cGAN_to_orig_cls_map_final')
        # modify each time:
        # here, inspect into images, and then manually run to modify our cGAN_to_orig_cls_map_final:
        # cGAN_to_orig_cls_map_final[xxx] = ...
        print()
        ### for Insects:
        #cGAN_to_orig_cls_map_final[116] = 69
        ### for Fungi:
        #cGAN_to_orig_cls_map_final[11] = 0
        ### for Birds:
        #cGAN_to_orig_cls_map_final[120] = 232
        #cGAN_to_orig_cls_map_final[117] = 271
        #cGAN_to_orig_cls_map_final[82] = 282
        #cGAN_to_orig_cls_map_final[96] = 315
        ### for Reptiles:
        cGAN_to_orig_cls_map_final[19] = 181
        cGAN_to_orig_cls_map_final[36] = 189
        cGAN_to_orig_cls_map_final[11] = 197
    
    assert(len(cGAN_to_orig_cls_map_final) == len(cls_folders))
    
    # save cGAN_to_orig_cls_map_final to pkl:
    f_pkl = open(rootDir+'cGAN_to_orig_cls_map_final.pkl', 'wb')
    pickle.dump(cGAN_to_orig_cls_map_final,f_pkl)
    f_pkl.close()
        


