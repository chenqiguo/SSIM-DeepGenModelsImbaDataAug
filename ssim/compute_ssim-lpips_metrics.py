#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:27:42 2024

@author: ps
"""

# here the metric is ssim/lpips value!!!


from skimage.metrics import structural_similarity as ssim
import cv2
import argparse
import os
import numpy as np
import random
import lpips


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



def get_img_ssim(file):
    img0 = cv2.imread(file)
    #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0 = center_crop_func(img0)
    
    return img0


def get_img_lpips(file):
    cv2_img0 = lpips.load_image(file)
    cent_crop_cv2_img0 = center_crop_func(cv2_img0)
    img0 = lpips.im2tensor(cent_crop_cv2_img0)
    #if(opt.use_gpu):
    img0 = img0.cuda(opt.gpu)
    
    return img0




if __name__ == "__main__":
    opt = parser.parse_args()
    
    ## Initializing the model for lpips:
    loss_fn = lpips.LPIPS(net='alex',version=opt.version)
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
    avg_scores_list = []
    for i in img_pth_dict.keys(): # for each super-class
        #if i not in img_pth_dict:
        #    continue
        files = img_pth_dict[i]
        random.shuffle(files)
        if(opt.N is not None):
        	files = files[:opt.N]
        
        scores = []
        #dists = []
        for (ff,file) in enumerate(files[:-1]):
            #print('***** debug: ' + file)
            
            img0_ssim = get_img_ssim(file)
            img0_lpips = get_img_lpips(file)
            
            #print('**** debug: img0_ssim.shape = ' + str(img0_ssim.shape)) # (32, 32, 3)
            #assert(False)
            
            if(opt.all_pairs):
                files1 = files[ff+1:]
            else:
                files1 = [files[ff+1],]
        
            for file1 in files1:
                img1_ssim = get_img_ssim(file1)
                img1_lpips = get_img_lpips(file1)
                
                # Compute SSIM score and distance between two images:
                (score, diff) = ssim(img0_ssim, img1_ssim, full=True, channel_axis=2)
                print('ssim (%s,%s): %.3f'%(file,file1,score))
                
                # Compute LPIPS distance
                dist01 = loss_fn.forward(img0_lpips,img1_lpips)
                print('lpips (%s,%s): %.3f'%(file,file1,dist01))
                
                this_score = score/ max(1e-8,dist01.item())
                
                scores.append(this_score)
                
        avg_scores = np.mean(np.array(scores))
        stderr_scores = np.std(np.array(scores))/np.sqrt(len(scores))
        avg_scores_list.append(avg_scores)
        
        print('Avg: %.5f +/- %.5f'%(avg_scores,stderr_scores))
        f_out.writelines('For super-class %d, Avg ssim/lpips score: %.6f +/- %.6f\n'%(i, avg_scores,stderr_scores))
        
        
        #print()
    
    
    final_score_max = max(avg_scores_list)
    final_score_mean = np.mean(avg_scores_list)
    
    print('--------- Max: %.5f'%(final_score_max))
    print('--------- Mean: %.5f'%(final_score_mean))
    f_out.writelines('--------- For all super-classes above, Final ssim/lpips score (the max): %.6f\n'%(final_score_max))
    f_out.writelines('--------- For all super-classes above, Final ssim/lpips score (the mean): %.6f'%(final_score_mean))
    
    
    f_out.close()
            






