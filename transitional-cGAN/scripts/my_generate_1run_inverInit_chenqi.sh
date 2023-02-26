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

####################################################################################################

## for Birds step1:
CUDA_VISIBLE_DEVICES=1 python generate_1run_inverInit_chenqi.py \
 --outdir=results/project/Birds_128-011200/ \
 --classes_orig=267,327 \
 --target_root=datasets/Birds/ \
 --network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl \
 --map_pkl_dir=datasets/Birds_128_map/cGAN_to_orig_cls_map_final.pkl




#202-327:
#215,223,228,235,237,239,243,244,245,246,247,248,249,250,252,253,254,255,257,258,259 -> 7d GPU0 (no tmp)
#260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280 -> 7d GPU1 (tmp)

#281,282,283,284,285,286,287,288,290,291,292,293,294,295,296,297,298,299,300,301,302 -> 8d GPU0 (tmp2)       
#303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,320,322,323,324,325,326,327 -> 8d GPU1 (tmp3)          



# -> Titan GPU0 (tmp4)    NOT use yet!!!            









