#!/bin/bash

# on each super-class:

# for iNaturalist dataset:
python cls_res.py -a resnet18 --gpu 0 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Insects \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_orig_iNaturalist/Insects
 
python cls_res.py -a resnet18 --gpu 0 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Birds \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_orig_iNaturalist/Birds
 
python cls_res.py -a resnet18 --gpu 0 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Reptiles \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_orig_iNaturalist/Reptiles
 
python cls_res.py -a resnet18 --gpu 0 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Fungi \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_orig_iNaturalist/Fungi
 
python cls_res.py -a resnet18 --gpu 0 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Amphibians \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_orig_iNaturalist/Amphibians


python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Amphibians \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Amphibians


# for flowers dataset:
python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/flowers \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/flowers/cls_res18_orig


# for UTKFace dataset:
python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/UTKFace/UTKFace/utkface_aligned_cropped/group_by_ethnicity \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/UTKFace/cls_res18_orig


# for scene dataset:
python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene/scene \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/cls_res18_orig



## for orig iNatruarlist (all except Plants):
python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allButPlants







# on all sub-classes together:

python cls_res.py -a resnet18 --gpu 0 --epochs 100 -b 128 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_allSubCls \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/cls_res18_orig_iNaturalist \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/cls_res18_orig_iNaturalist/part4/checkpoint_bestAcc1.pth.tar

# on all super-classes:
python cls_res.py -a resnet18 --gpu 1 --epochs 100 -b 128 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_allSuperCls \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSuperCls/cls_res18_orig_iNaturalist \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSuperCls/cls_res18_orig_iNaturalist/part4/checkpoint_bestAcc1.pth.tar

# to resume:
#--resume





