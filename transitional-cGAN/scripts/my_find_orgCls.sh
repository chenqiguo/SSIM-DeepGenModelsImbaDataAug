#!/bin/bash
#source ~/envs/ada/bin/activate; module load cuda/11.2.2 ninja

### our cmds:

## for Insects:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 1  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Insects_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Insects_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Insects/checkpoint_bestAcc1.pth.tar

## for Fungi:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 1  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Fungi_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Fungi_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Fungi/checkpoint_bestAcc1.pth.tar

## for Birds:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 0  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Birds_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Birds_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Birds/checkpoint_bestAcc1.pth.tar

## for Reptiles:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 1  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Reptiles_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Reptiles_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Reptiles/checkpoint_bestAcc1.pth.tar

## for Amphibians:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 0  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Amphibians_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Amphibians_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Amphibians/step1_v2/thresh_12/ckpt/checkpoint_bestAcc1.pth.tar
#/eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_orig_iNaturalist/Amphibians/ckpt/checkpoint_bestAcc1.pth.tar



CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Insects_128-007000/140 --trunc=1 --class=140 --seeds=600-3600 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

## for Fungi:
# seeds=700-5700, for class 0-11 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Fungi_128-006800/11 --trunc=1 --class=11 --seeds=700-5700 \
--network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl



## for flowers:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 0  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/flowers_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/flowers_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/flowers/cls_res18_orig/ckpt/checkpoint_bestAcc1.pth.tar


## for UTKFace:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 1  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/UTKFace_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/UTKFace_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/UTKFace/cls_res18_orig/ckpt/checkpoint_bestAcc1.pth.tar


## for scene:
python find_orgCls_chenqi_step1.py -a resnet18 --gpu 1  \
 --test-data /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/scene_128 \
 --test-result /eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/scene_128_map \
 --network /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/cls_res18_orig/ckpt/checkpoint_bestAcc1.pth.tar









