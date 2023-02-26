#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:22:00 2021

@author: guo.1648
"""

# compute mean & std for cartoon_output training set.



from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2
from PIL import Image


def my_4ch_loader(filename):
    #return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return Image.open(filename)

def get_mean_std_forDataset(data_dir,batch_size,isGray):
    # newly added: compute the mean and std for transforms.Normalize using whole dataset:
    tmp_data = datasets.ImageFolder(root=data_dir,
                                    #loader=my_4ch_loader, # for 4ch input imgs!
                                    transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.ToTensor()]))
    tmp_loader = DataLoader(tmp_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    if not isGray:
        #for data, _ in tmp_loader:
        for i, (data, _) in enumerate(tmp_loader):
            print(i)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
    #else: for MNIST
    
    mean /= nb_samples
    std /= nb_samples
    
    return (mean, std)




if __name__ == "__main__":
    
    data_dir = '/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Amphibians_2ndTry/train/' #'/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/UTKFace/opt2/step2/thresh_25/train/' #'/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/flowers/'
    batch_size = 256
    
    img_means, img_stds = get_mean_std_forDataset(data_dir,batch_size,isGray=False)
    
    print('img_means = ' + str(img_means))
    print('img_stds = ' + str(img_stds))
    
    # results:
    
    
    
    
    

