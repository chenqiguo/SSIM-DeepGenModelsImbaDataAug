#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:20:01 2022

@author: guo.1648
"""

# referenced from imgSelection_chenqi.py

# using the all_prob_dict.pkl generated in imgSelectionMAE_chenqi.py, 
# plot&save pred_prob distribution hist for further analysis


import os
import pickle
import matplotlib.pyplot as plt


srcRootDir = 'cGAN_data/generate/'

supCls_folder = 'scene_128-004800/val/'

pklFileName = 'all_prob_dict.pkl'


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
    srcDir_val = srcRootDir + supCls_folder
    cls_folders = get_clsFolders(srcDir_val)
    
    subCls_num = len(cls_folders)
    
    # init dict to store all sub-class distributions:
    dist_dict = {}
    for cls_folder in cls_folders:
        dist_dict[cls_folder] = []
    
    for this_imgFullName in all_prob_dict:
        print(this_imgFullName)
        this_pred_prob = all_prob_dict[this_imgFullName]
        
        this_cls_folder = this_imgFullName.split('/')[-2]
        dist_dict[this_cls_folder].append(this_pred_prob)
        
    # plot and save the distribution histograms for each sub-class:
    for cls_folder in cls_folders:
        
        all_probVal_list = dist_dict[cls_folder]
        
        fig = plt.figure()         # create a figure instance
        ax = fig.add_subplot(111)   # and axes
        ax.hist(all_probVal_list, density=False, bins=30)  # density=False would make counts; True would make probability
        plt.ylabel('Count')
        plt.xlabel('Prediction probabilities');
        plt.title('cls'+ cls_folder + ' predicted as true') # false
        # plt.show()                # this would show the plot, but you can leave it out 
        
        # Save the figure to the current path
        plt.savefig(srcRootDir+supCls_folder+"predTrue_prob_cls"+cls_folder+".png") # predFalse_prob_cls



