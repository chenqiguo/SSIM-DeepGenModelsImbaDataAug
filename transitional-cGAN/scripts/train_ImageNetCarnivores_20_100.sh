#!/bin/bash
#source ~/envs/ada/bin/activate; module load cuda/11.2.2 ninja

python train.py --outdir=results/training-runs \
--data=datasets/ImageNet_Carnivores_20_100.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full


### our cmds:

python train.py --outdir=results/training-runs \
--data=datasets/Fungi_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full

python train.py --outdir=results/training-runs \
--data=datasets/Insects_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full

python train.py --outdir=results/training-runs \
--data=datasets/Birds_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full

python train.py --outdir=results/training-runs \
--data=datasets/Reptiles_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full
#--resume=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/training-runs/00004-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-005800.pkl

python train.py --outdir=results/training-runs \
--data=datasets/Amphibians_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full

python train.py --outdir=results/training-runs \
--data=datasets/flowers_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full

python train.py --outdir=results/training-runs \
--data=datasets/UTKFace_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full

python train.py --outdir=results/training-runs \
--data=datasets/scene_128.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=1 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full 
#--resume=/eecf/cbcsl/data100b/Chenqi/transitional-cGAN/results/training-runs/00010-scene_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-004800.pkl
















