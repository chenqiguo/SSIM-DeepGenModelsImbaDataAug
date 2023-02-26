#!/bin/bash
#source ~/envs/ada/bin/activate; module load cuda/11.2.2 ninja

"""Generate images using pretrained network pickle.
Chenqi's version: generate images in all needed sub-classes in one run.
"""



### our cmds:




####################################################################################################

## for Reptiles step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/Reptiles_128-011200/val \
 --trunc=1 --seeds=8586015-9586015 \
 --classes_orig=199 \
 --network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Reptiles_128_map/cGAN_to_orig_cls_map_final.pkl


## for Reptiles step2:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/Reptiles_128-011200_step2_tmp4/val \
 --trunc=1 --seeds=15016022-17016022 \
 --classes_orig=201 \
 --network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Reptiles_128_map/cGAN_to_orig_cls_map_final.pkl

#15016022-17016022
#17016023-19016023

## for Reptiles without_cls_select:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Reptiles \
 --trunc=1 --seeds=1-2770 \
 --classes_orig=163-201 \
 --network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Reptiles_128_map/cGAN_to_orig_cls_map_final.pkl

CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Reptiles_2ndTry \
 --trunc=1 --seeds=1-2770 \
 --classes_orig=163-201 \
 --network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Reptiles_128_map/cGAN_to_orig_cls_map_final.pkl


####################################################################################################

## for Insects step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/Insects_128-007000_tmp3/val \
 --trunc=1 --seeds=4700015-5300015 \
 --classes_orig=75,76,77,85,86,96,98 \
 --network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Insects_128_map/cGAN_to_orig_cls_map_final.pkl

#12-152:
#24,25,26,28,30,33,34,35 -> 7d GPU0 (no tmp)
#36,37,38,46,47,49,50 -> 7d GPU1 (tmp)

#62,63,65,71,73,74 -> 8d GPU0 (tmp2)       
#75,76,77,85,86,96,98 -> 8d GPU1 (tmp3)          

#52,53,54,55,58,59 -> Titan GPU0 (tmp4)    (5000015-5600015): NOT run yet!!!      


## for Insects without_cls_select:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Insects \
 --trunc=1 --seeds=1-993 \
 --classes_orig=12-152 \
 --network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Insects_128_map/cGAN_to_orig_cls_map_final.pkl
      

####################################################################################################

## for Birds step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/Birds_128-011200_tmp/val \
 --trunc=1 --seeds=3050010-3450010 \
 --classes_orig=285,286,287,288,290,292,293,294,295 \
 --network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Birds_128_map/cGAN_to_orig_cls_map_final.pkl

#202-327:
#268,269,271,274,280,281,282,283,284 -> 7d GPU0 (no tmp)
#285,286,287,288,290,292,293,294,295 -> 7d GPU1 (tmp)

#296,297,299,300,301,302,303,304,305 -> 8d GPU0 (tmp2)       
#306,307,308,309,310,311,312,314,315,316 -> 8d GPU1 (tmp3)          

## for Birds without_cls_select:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Birds \
 --trunc=1 --seeds=1-1121 \
 --classes_orig=202-327 \
 --network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Birds_128_map/cGAN_to_orig_cls_map_final.pkl


####################################################################################################

## for Amphibians step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/Amphibians_128-009200/val \
 --trunc=1 --seeds=100001-1100001 \
 --classes_orig=162 \
 --network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Amphibians_128_map/cGAN_to_orig_cls_map_final.pkl

#153-162:
#153,154,155,156,157,158,159,160,161,162 -> Titan GPU0 (no tmp)  


## for Amphibians without_cls_select:         
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Amphibians \
 --trunc=1 --seeds=1-2770 \
 --classes_orig=153-162 \
 --network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Amphibians_128_map/cGAN_to_orig_cls_map_final.pkl

# 2nd try:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Amphibians_2ndTry \
 --trunc=1 --seeds=1-2770 \
 --classes_orig=153-162 \
 --network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Amphibians_128_map/cGAN_to_orig_cls_map_final.pkl



####################################################################################################

## for Fungi step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/Fungi_128-006800/val \
 --trunc=1 --seeds=0-100000 \
 --classes_orig=0,1,2,3,4,5,6,7,8,9,10,11 \
 --network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Fungi_128_map/cGAN_to_orig_cls_map_final.pkl

#0-11:
#0,1,2,3,4,5,6,7,8,9,10,11 -> Titan GPU0 (no tmp)      


## for Fungi without_cls_select:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/Fungi \
 --trunc=1 --seeds=1-2770 \
 --classes_orig=0-11 \
 --network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/Fungi_128_map/cGAN_to_orig_cls_map_final.pkl
     


####################################################################################################

## for flowers step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/flowers_128-007600/val \
 --trunc=1 --seeds=10001-20001 \
 --classes_orig=2 \
 --network=results/training-runs/00008-flowers_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007600.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/flowers_128_map/cGAN_to_orig_cls_map_final.pkl

#0-4:
#0,1,2,3,4 -> Titan GPU0 (no tmp)   


## for flowers without_cls_select:  
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/flowers \
 --trunc=1 --seeds=1-2770 \
 --classes_orig=0-4 \
 --network=results/training-runs/00008-flowers_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007600.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/flowers_128_map/cGAN_to_orig_cls_map_final.pkl      


####################################################################################################

## for UTKFace step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/UTKFace_128-004200/val \
 --trunc=1 --seeds=50001-150001 \
 --classes_orig=1 \
 --network=results/training-runs/00009-UTKFace_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-004200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/UTKFace_128_map/cGAN_to_orig_cls_map_final.pkl

#0-4:
#0,1,2,3,4 -> Titan GPU0 (no tmp)           


## for UTKFace without_cls_select:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/UTKFace \
 --trunc=1 --seeds=1-10000 \
 --classes_orig=0-4 \
 --network=results/training-runs/00009-UTKFace_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-004200.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/UTKFace_128_map/cGAN_to_orig_cls_map_final.pkl


####################################################################################################

## for scene step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/mae/cGAN_data/generate/scene_128-004800/val \
 --trunc=1 --seeds=0-20000 \
 --classes_orig=0,1,2,3,4,5 \
 --network=results/training-runs/00010-scene_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-004800.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/scene_128_map/cGAN_to_orig_cls_map_final.pkl

#0-4:
#0,1,2,3,4,5 -> Titan GPU0 (no tmp)           

## for scene without_cls_select:
CUDA_VISIBLE_DEVICES=1 python generate_1run_chenqi.py \
 --outdir_root=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/generate_without_cls_select/scene \
 --trunc=1 --seeds=1-3000 \
 --classes_orig=0-5 \
 --network=results/training-runs/00010-scene_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-004800.pkl \
 --map_pkl_dir=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/datasets/scene_128_map/cGAN_to_orig_cls_map_final.pkl




















