#!/bin/bash

# on each super-class:

python cls_res_test.py -a resnet18 --gpu 0  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3/iNaturalist_eachSubCls/Insects/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Insects/checkpoint_bestAcc1.pth.tar

python cls_res_test.py -a resnet18 --gpu 0  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3/iNaturalist_eachSubCls/Fungi/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Fungi/checkpoint_bestAcc1.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Fungi/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug3 \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug3/ckpt/checkpoint_Epoch100.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Insects/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step1 \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step1/ckpt/checkpoint_Epoch100.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Insects/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step2 \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step2/ckpt/checkpoint_bestAcc1.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Insects/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step3 \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step3/part3/checkpoint_bestAcc1.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Birds/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Birds \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Birds/checkpoint_bestAcc1.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Amphibians/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Amphibians \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_iNaturalist/Amphibians/checkpoint_bestAcc1.pth.tar

python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/data/iNaturalist_eachSubCls/Reptiles/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Reptiles \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Reptiles/checkpoint_bestAcc1.pth.tar


python cls_res_test.py -a resnet18 --gpu 0  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/flowers/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/flowers/cls_res18_orig \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/flowers/cls_res18_orig/ckpt/checkpoint_bestAcc1.pth.tar


python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/UTKFace/UTKFace/utkface_aligned_cropped/group_by_ethnicity/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/UTKFace/cls_res18_orig \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/UTKFace/cls_res18_orig/ckpt/checkpoint_bestAcc1.pth.tar


python cls_res_test.py -a resnet18 --gpu 1  \
 --test-data=/eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene/scene/train \
 --test-result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/cls_res18_orig \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/cls_res18_orig/ckpt/checkpoint_bestAcc1.pth.tar







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





