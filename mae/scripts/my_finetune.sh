

### Script for ViT-Huge:

python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 --use_volta32 \
    --batch_size 16 \
    --model vit_huge_patch14 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.3 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}

#########################################################################################

### our cmd:
### for cls_resnet18:

## for orig Fungi: 2568
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Fungi/ --nb_classes 12

## for cGAN-aug Fungi: 19474
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Fungi_s2t10/ --nb_classes 12

## for orig Insects: 44766
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Insects/ --nb_classes 141

## for cGAN-aug Insects: 17123
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Insects_s2t10/ --nb_classes 141 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/17123/checkpoint-48.pth

## for orig Birds: 11528
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Birds/ --nb_classes 126

## for cGAN-aug Birds: 68807
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Birds_s1t10/ --nb_classes 126 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/68807/checkpoint-47.pth


## for orig Reptiles: 15197
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Reptiles/ --nb_classes 39

## for cGAN-aug Reptiles: 23522
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Reptiles_s2t20/ --nb_classes 39


## for orig Amphibians: 11179
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Amphibians/ --nb_classes 10

## for cGAN-aug Amphibians: 26992
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/Amphibians_s1t10/ --nb_classes 10


## for orig flowers: 51956
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_resnet18/flowers/ --nb_classes 5


## for orig UTKFace: 21569
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_resnet18/UTKFace/ --nb_classes 5


## for orig scene: 39452
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_resnet18/scene/ --nb_classes 6



#########################################################################################

### our cmd:
### for cls_mae:

## for cGAN-aug Reptiles step1: 25931
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Reptiles_s1t3/ --nb_classes 39

## for cGAN-aug Reptiles step2: 12945
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Reptiles_s2t5/ --nb_classes 39

## Fabian's experiment: resume the MAE_orig classifier on cGAN-aug Reptiles dataset for 10 epochs: --> These NOT work!!!
# resume on last epoch with step2 data: 33664
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 60 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Reptiles_s2t5/ --nb_classes 39 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/15197/checkpoint-49.pth
# resume on best epoch with step2 data: 30589
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 33 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Reptiles_s2t5/ --nb_classes 39 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/15197/checkpoint-23.pth
# resume on last epoch with step1 data: 33986
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 60 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Reptiles_s1t3/ --nb_classes 39 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/15197/checkpoint-49.pth
# resume on best epoch with step1 data: 65392
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 33 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Reptiles_s1t3/ --nb_classes 39 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/15197/checkpoint-23.pth


## for cGAN-aug Insects step1 (for debug: NOT finish cGAN-aug): 45190
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Insects_s1_debug/ --nb_classes 141


## for cGAN-aug Amphibians step1 thresh2: 59826
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Amphibians_s1t2/ --nb_classes 10
## for cGAN-aug Amphibians step1 thresh3: 48467
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Amphibians_s1t3/ --nb_classes 10


## for cGAN-aug Fungi step1 thresh3: 20242
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Fungi_s1t3/ --nb_classes 12


## for cGAN-aug Birds: 32327
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/Birds_s1t3/ --nb_classes 126



## for cGAN-aug flowers step1 thresh2: 61063
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/flowers_s1t2/ --nb_classes 5


## for cGAN-aug UTKFace step1 thresh2: 31531
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/UTKFace_s1t2/ --nb_classes 5


## for cGAN-aug scene step1 thresh2.5: 51837
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/scene_s1t2.5/ --nb_classes 6


#########################################################################################
## on the Whole iNatruarlist dataset (both original & after cGAN-MAEcls augmented)

## for orig iNatruarlist (all except Plants): 27810
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_resnet18/iNaturalist/ --nb_classes 328

## for cGAN-aug iNatruarlist (all except Plants): resume: 4664
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/cls_mae/iNaturalist_cGANaug/ --nb_classes 328 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/75094/checkpoint-44.pth



#########################################################################################
## for the cGAN-aug dataset without_cls_select:

## for Amphibians: 11335
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/Amphibians/ --nb_classes 10

## for Insects: 32124
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/Insects/ --nb_classes 141 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/4108/checkpoint-47.pth

## for Reptiles: 6187
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/Reptiles/ --nb_classes 39

## for Fungi: 45038
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/Fungi/ --nb_classes 12

## for Birds: 15575; resume: 8848
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/Birds/ --nb_classes 126 \
    --resume /eecf/cbcsl/data100b/Chenqi/mae/checkpoint/guo.1648/experiments/15575/checkpoint-37.pth

## for flowers: 63308
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/flowers/ --nb_classes 5

## for scene: 29687
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/scene/ --nb_classes 6

## for UTKFace: 44982
python submitit_finetune.py \
    --ngpus 1 \
    --nodes 1 \
    --batch_size 8 \
    --model vit_huge_patch14 \
    --finetune pretrained_ckpt/mae_pretrain_vit_huge.pth \
    --epochs 50 --accum_iter 8 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path dataset/without_cls_select/UTKFace/ --nb_classes 5













