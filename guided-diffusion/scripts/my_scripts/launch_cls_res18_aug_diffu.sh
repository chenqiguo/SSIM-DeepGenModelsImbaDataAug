#!/bin/bash

## on each super-class:


# for Fungi:

python /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/for_resnet/Fungi/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/Fungi/step1/thresh_10

python /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/for_resnet/Fungi_new/step1/thresh_10  \
 --result /eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/Fungi_new/step1/thresh_10 


# for Birds:

python /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/for_resnet/Birds/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/Birds/step1/thresh_10


# for scene:

python /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/for_resnet/scene/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/scene/step1/thresh_10


# for Amphibians:

python /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/for_resnet/Amphibians/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/guided-diffusion/results/cls_res18_aug_diffu/Amphibians/step1/thresh_10





  
# to resume:
# --resume





