#!/bin/bash

# on each super-class:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3/iNaturalist_eachSubCls/Insects \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3_iNaturalist/Insects

python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3/iNaturalist_eachSubCls/Fungi \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3_iNaturalist/Fungi







python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_eachSubCls/Birds \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_gan_iNaturalist/Birds
 
python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_eachSubCls/Reptiles \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_gan_iNaturalist/Reptiles
  
python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_eachSubCls/Amphibians \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_gan_iNaturalist/Amphibians \
 
# on all sub-classes together:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 -b 128 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_allSubCls \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/cls_res18_gan_iNaturalist \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/cls_res18_gan_iNaturalist/part5/checkpoint_bestAcc1.pth.tar

# on all super-classes:

python cls_res.py -a resnet18 --gpu 0 --epochs 40 -b 128 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_allSuperCls \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSuperCls/cls_res18_gan_iNaturalist \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSuperCls/cls_res18_gan_iNaturalist/part1/checkpoint_bestAcc1.pth.tar


  
# to resume:
# --resume





