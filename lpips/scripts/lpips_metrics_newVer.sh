#!/bin/bash

# on each super-class:

# for CIFAR-100 dataset (with 20 coarse labels):
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_coarse.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100_coarse_lpips.txt 

# for CIFAR-100 imb100 dataset (with 20 coarse labels):
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_imb100Coarse.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100imb100_coarse_lpips.txt 

# for ImageNet dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_coarse/train_paths_coarse.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet_coarse_lpips.txt 

# for ImageNet-LT dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_coarse/ImageNet_LT_train_coarse.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet-LT_coarse_lpips.txt 

# for coarse VOC dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/train_paths_coarse.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages \
 --out results/VOC_coarse_lpips.txt 

# for original Adience age dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/Adience/datasets/age_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/Adience/datasets/age \
 --out results/Adience_coarse_lpips.txt

# for original flowers dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers/train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers \
 --out results/flowers_coarse_lpips.txt

# for original scene dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned/train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned \
 --out results/scene_coarse_lpips.txt

# for coarse iNaturalist-Birds dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019/Birds_train_coarse_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Birds_coarse_lpips.txt 

# for coarse iNaturalist-Insects dataset:
python compute_lpips_metrics_newVer.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019/Insects_train_coarse_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Insects_coarse_lpips.txt 









