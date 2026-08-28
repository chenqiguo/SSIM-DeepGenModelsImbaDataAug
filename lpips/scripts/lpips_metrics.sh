#!/bin/bash

# on each super-class:



# for flowers dataset:
python compute_lpips_metrics.py --gpu 0 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/flowers/train/4 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/flowers/4_dists_pair.txt


# for UTKFace dataset:
python compute_lpips_metrics.py --gpu 0 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/UTKFace/cls_by_race/train/4 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/UTKFace/4_dists_pair.txt


# for scene dataset:
python compute_lpips_metrics.py --gpu 0 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/scene/cleaned/train/5 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/scene/5_dists_pair.txt


# for iNaturalist dataset:
python compute_lpips_metrics.py --gpu 3 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Amphibians/sup_4 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Amphibians/sup4_dists_pair.txt

python compute_lpips_metrics.py --gpu 3 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Fungi/sup_4 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Fungi/sup4_dists_pair.txt

python compute_lpips_metrics.py --gpu 3 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Reptiles/sup_2 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Reptiles/sup2_dists_pair.txt

python compute_lpips_metrics.py --gpu 3 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Birds/sup_9 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Birds/sup9_dists_pair.txt

python compute_lpips_metrics.py --gpu 3 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Insects/sup_14 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Insects/sup14_dists_pair.txt





python compute_lpips_metrics.py --gpu 0 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Fungi/sup_1 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Fungi/sup1_dists_pair.txt --all-pairs

python compute_lpips_metrics.py --gpu 1 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Fungi/sup_2 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Fungi/sup2_dists_pair.txt --all-pairs

python compute_lpips_metrics.py --gpu 2 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Fungi/sup_3 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Fungi/sup3_dists_pair.txt --all-pairs

python compute_lpips_metrics.py --gpu 3 \
 -d /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/data/inaturalist-2019_supCls/train/Fungi/sup_4 \
 -o /home/ps/scratch/SSIM-DeepGenModelsImbaDataAug/lpips/results/inaturalist-2019_supCls/Fungi/sup4_dists_pair.txt --all-pairs






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





