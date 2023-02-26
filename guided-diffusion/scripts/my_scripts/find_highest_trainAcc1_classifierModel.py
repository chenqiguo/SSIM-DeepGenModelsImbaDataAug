#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:44:21 2022

@author: guo.1648
"""

# From the progress.csv file during training,
# find the best classifier / diffusion model which has the highest val_acc@1 value / lowest loss value.
# Also plot the learning curve of val_acc@1 / loss for visualization.


import pandas as pd
import matplotlib.pyplot as plt


#"""
### (1) for classifier:

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/classifier_diffusion/Amphibians/classifier/'
csvFileName = 'progress.csv'


data = pd.read_csv (srcRootDir + csvFileName)   
df = pd.DataFrame(data, columns= ['train_acc@1']) #val_acc@1
#print (df)
print(df.idxmax()) # get index of max value
# df.iloc[6000] ... --> find the suitable pt one by one!

df.plot(y='train_acc@1', use_index=True, kind = 'line') #val_acc@1
#plt.show()
plt.savefig(srcRootDir + 'progress.png')
#"""

"""
### (2) for diffusion model:

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/classifier_diffusion/Amphibians/diffusion_model_resumed/'
csvFileName = 'progress.csv'


data = pd.read_csv (srcRootDir + csvFileName)   
df = pd.DataFrame(data, columns= ['loss'])
#print (df)


df.plot(y='loss', use_index=True, kind = 'line')
#plt.show()
plt.savefig(srcRootDir + 'progress.png')

#print(df.min())
print(df.idxmin()) # get index of max value
# df.iloc[6000] ... --> find the suitable pt one by one!

"""




