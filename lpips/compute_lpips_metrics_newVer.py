#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:08:19 2023

@author: ps
"""


# similar to the implementation of our SSIM-supSubCls metric.
# here we use LPIPS instead.

# referenced from:
# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_1dir_allpairs.py


import lpips
#import torch
import cv2
import argparse
import os
import numpy as np
import random

# default img size for computing LPIPS:
new_img_size = 32 #64 # 64x64 RGB img

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--txt', type=str, 
                    default='/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_coarse.txt', 
                    help='fullname of txt file which saves the training image paths and class labels')
parser.add_argument('--root', type=str, 
                    default='/home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img',
                    help='root dir in which saves all the images')
parser.add_argument('--out', type=str, 
                    default='results/CIFAR100_coarse_ssim.txt',
                    help='')
parser.add_argument('--all-pairs', action='store_true', help='turn on to test all N(N-1)/2 pairs, leave off to just do consecutive pairs (N-1)')
parser.add_argument('-N', type=int, default=None)
parser.add_argument('-v','--version', type=str, default='0.1')
#parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')





"""
# Quick start example:
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptial loss, when used for optimization

img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
"""




def center_crop_func(img):
    # Note: here the img acquired from lpips load_image has the channel inverted
    # for the last dim: e.g., img with shape (375,500,3) is RGB instead of BGR!
    
    width, height = img.shape[1], img.shape[0]
    # crop to be square:
    crop_dim = min(width, height)
    
    mid_x, mid_y = int(width/2), int(height/2)
    half_cd = int(crop_dim/2)
    
    crop_img = img[mid_y-half_cd:mid_y+half_cd, mid_x-half_cd:mid_x+half_cd]
    
    cent_crop_img = cv2.resize(crop_img, (new_img_size,new_img_size))
    
    return cent_crop_img




if __name__ == "__main__":
    
    
    opt = parser.parse_args()
    
    ## Initializing the model
    loss_fn = lpips.LPIPS(net='vgg',version=opt.version) # alex
    #if(opt.use_gpu):
    loss_fn.cuda(opt.gpu)
    
    # load the image paths:
    #total_supCls_num = 0
    img_pth_dict = {}
    
    with open(opt.txt) as f:
        for line in f:
            this_label = int(line.split()[1])
            this_img_path = os.path.join(opt.root, line.split()[0])
            if this_label not in img_pth_dict:
                img_pth_dict[this_label] = [this_img_path]
                #total_supCls_num += 1
            else:
                img_pth_dict[this_label].append(this_img_path)
    
    
    f_out = open(opt.out,'w')
    avg_dists_list = []
    for i in img_pth_dict.keys(): # for each super-class
        #if i not in img_pth_dict:
        #    continue
        files = img_pth_dict[i]
        random.shuffle(files)
        if(opt.N is not None):
        	files = files[:opt.N]
        
        #scores = []
        dists = []
        for (ff,file) in enumerate(files[:-1]):
            
            print('***** debug: ' + file)
            
            cv2_img0 = lpips.load_image(file)
            cent_crop_cv2_img0 = center_crop_func(cv2_img0)
            img0 = lpips.im2tensor(cent_crop_cv2_img0)
            #if(opt.use_gpu):
            img0 = img0.cuda(opt.gpu)
            
            if(opt.all_pairs):
                files1 = files[ff+1:]
            else:
                files1 = [files[ff+1],]
        
            for file1 in files1:
        		#img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir,file1)))
                cv2_img1 = lpips.load_image(file1)
                cent_crop_cv2_img1 = center_crop_func(cv2_img1)
                img1 = lpips.im2tensor(cent_crop_cv2_img1)
        		#if(opt.use_gpu):
                img1 = img1.cuda(opt.gpu)
        
        		# Compute distance
                dist01 = loss_fn.forward(img0,img1)
                print('(%s,%s): %.3f'%(file,file1,dist01))
                
                dists.append(dist01.item())
        
        avg_dists = np.mean(np.array(dists))
        stderr_dists = np.std(np.array(dists))/np.sqrt(len(dists))
        avg_dists_list.append(avg_dists)
        
        print('Avg: %.5f +/- %.5f'%(avg_dists,stderr_dists))
        f_out.writelines('For super-class %d, Avg lpips dist: %.6f +/- %.6f\n'%(i, avg_dists,stderr_dists))
        
        
        #print()
    
    
    final_lpips = min(avg_dists_list)
    mean_lpips = np.mean(avg_dists_list)
    
    print('--------- Min: %.5f'%(final_lpips))
    f_out.writelines('--------- For all super-classes above, Final lpips dist (the min): %.6f\n'%(final_lpips))
    
    print('--------- Mean: %.5f'%(mean_lpips))
    f_out.writelines('--------- For all super-classes above, Mean lpips dist (the mean): %.6f'%(mean_lpips))
    
    
    f_out.close()
    








