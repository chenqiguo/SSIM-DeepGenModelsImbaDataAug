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
new_img_size = 64 # 64x64 RGB img

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir_pair')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--all-pairs', action='store_true', help='turn on to test all N(N-1)/2 pairs, leave off to just do consecutive pairs (N-1)')
parser.add_argument('-N', type=int, default=None)
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
    
    """
    #### just a test for debug!:
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda(0)
    
    
    # newly modified by Chenqi: center crop & resize img0 and img1 to get same img size:
    cv2_img0 = lpips.load_image('/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers/val/1/10919961_0af657c4e8.jpg')
    cent_crop_cv2_img0 = center_crop_func(cv2_img0)
    img0 = lpips.im2tensor(cent_crop_cv2_img0)
    
    cv2_img1 = lpips.load_image('/home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers/val/1/11545123_50a340b473_m.jpg')
    cent_crop_cv2_img1 = center_crop_func(cv2_img1)
    img1 = lpips.im2tensor(cent_crop_cv2_img1)
    
    
    img0 = img0.cuda(0)
    img1 = img1.cuda(0)
    
    dist01 = loss_fn.forward(img0, img1)
    print('Distance: %.3f'%dist01)
    """
    
    ### USE BELOW CODE!
    
    opt = parser.parse_args()
    
    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex',version=opt.version)
    #if(opt.use_gpu):
    loss_fn.cuda(opt.gpu)
    
    # crawl directories
    f = open(opt.out,'w')
    files = os.listdir(opt.dir)
    random.shuffle(files)# newly modified by Chenqi: shuffle the files!
    if(opt.N is not None):
    	files = files[:opt.N]
    #F = len(files)
    
    dists = []
    for (ff,file) in enumerate(files[:-1]):
    	#img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir,file))) # RGB image from [-1,1]
        cv2_img0 = lpips.load_image(os.path.join(opt.dir,file))
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
            cv2_img1 = lpips.load_image(os.path.join(opt.dir,file1))
            cent_crop_cv2_img1 = center_crop_func(cv2_img1)
            img1 = lpips.im2tensor(cent_crop_cv2_img1)
    		#if(opt.use_gpu):
            img1 = img1.cuda(opt.gpu)
    
    		# Compute distance
            dist01 = loss_fn.forward(img0,img1)
            print('(%s,%s): %.3f'%(file,file1,dist01))
            f.writelines('(%s,%s): %.6f\n'%(file,file1,dist01))
    
            dists.append(dist01.item())
    
    avg_dist = np.mean(np.array(dists))
    stderr_dist = np.std(np.array(dists))/np.sqrt(len(dists))
    
    print('Avg: %.5f +/- %.5f'%(avg_dist,stderr_dist))
    f.writelines('Avg: %.6f +/- %.6f'%(avg_dist,stderr_dist))
    
    f.close()








