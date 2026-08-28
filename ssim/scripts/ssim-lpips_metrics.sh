#!/bin/bash

# on each super-class:

# for CIFAR-100 dataset (with 20 coarse labels):
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_coarse.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py --all-pairs \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100_fine_ssim.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/cifar-20/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100_SCANcoarse_ssim.txt

# for CIFAR-100 imb100 dataset (with 20 coarse labels):
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_imb100Coarse.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100imb100_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py --all-pairs \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/CIFAR100_train_imb100.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100imb100_fine_ssim.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/cifar-20imb/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100imb100_SCANcoarse_ssim.txt

# for CIFAR-100 dataset (with 4 coarse labels):
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_4superCls.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100_4superCls_ssim.txt 

# for CIFAR-100 imb100 dataset (with 4 coarse labels):
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/train_paths_imb100_4superCls.txt \
 --root /home/ps/scratch/KD_imbalance/BalancedKnowledgeDistillation/data/cifar-100-python/clean_img \
 --out results/CIFAR100imb100_4superCls_ssim.txt 
 
# for ImageNet dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_coarse/train_paths_coarse.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet/ImageNet_train.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet_fine_ssim.txt 
 
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/imagenet_51/predCls/SCAN_pred_val_paths.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet_SCANcoarse_ssim.txt

# for ImageNet-LT dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_coarse/ImageNet_LT_train_coarse.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet-LT_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/KD_imbalance/LFME/my_data/ImageNet_LT/ImageNet_LT_train.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet-LT_fine_ssim.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/imagenetLT_51/predCls/SCAN_pred_val_paths.txt \
 --root /home/ps/scratch/KD_imbalance/LFME/my_data/ILSVRC/Data/CLS-LOC \
 --out results/ImageNet-LT_SCANcoarse_ssim.txt

# for coarse VOC dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/train_paths_coarse.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages \
 --out results/VOC_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/train_paths_fine.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages \
 --out results/VOC_fine_ssim.txt 
 
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/voc/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/VOC/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages \
 --out results/VOC_SCANcoarse_ssim.txt 

# for original Adience age dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/Adience/datasets/age_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/Adience/datasets/age \
 --out results/Adience_coarse_ssim-lpips.txt

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/adience/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/Adience/datasets/age \
 --out results/Adience_SCANcoarse_ssim.txt

# for original flowers dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers/train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers \
 --out results/flowers_coarse_ssim-lpips.txt

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/flowers/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers \
 --out results/flowers_SCANcoarse_ssim.txt

# for original scene dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned/train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned \
 --out results/scene_coarse_ssim-lpips.txt

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/scene/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned \
 --out results/scene_SCANcoarse_ssim.txt


# for coarse iNaturalist-Birds dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019/Birds_train_coarse_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Birds_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019/Birds_0startLabels_train_fine_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Birds_fine_ssim.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/birds/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Birds_SCANcoarse_ssim.txt


# for coarse iNaturalist-Insects dataset:
python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019/Insects_train_coarse_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Insects_coarse_ssim-lpips.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019/Insects_0startLabels_train_fine_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Insects_fine_ssim.txt 

python compute_ssim-lpips_metrics.py \
 --txt /home/ps/scratch/SCAN_Unsupervised-Classification-master/results/debug/insects/predCls/SCAN_pred_train_paths.txt \
 --root /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019 \
 --out results/Insects_SCANcoarse_ssim.txt










