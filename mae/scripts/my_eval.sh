

### Script for ViT-Huge:

python main_finetune.py --eval --resume mae_finetuned_vit_huge.pth --model vit_huge_patch14 --batch_size 16 --data_path ${IMAGENET_DIR}


### our cmd:

####################################################################################################
## for Reptiles:
## for step1: using orig Reptiles MAE classifider to select images: 15197
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 39 \
    --data_path cGAN_data/generate/Reptiles_128-011200/ \
    --resume checkpoint/guo.1648/experiments/15197/checkpoint-23.pth
## for step2: using step1 Reptiles MAE classifider to select images: 25931
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 39 \
    --data_path cGAN_data/generate/Reptiles_128-011200_step2_tmp4/ \
    --resume checkpoint/guo.1648/experiments/25931/checkpoint-45.pth

####################################################################################################
## for Insects:
## for step1: using orig Insects MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 141 \
    --data_path cGAN_data/generate/Insects_128-007000_tmp3/ \
    --resume checkpoint/guo.1648/experiments/44766/checkpoint-41.pth

#12-152
#Insects_128-007000/ -> 7d GPU0
#Insects_128-007000_tmp/ -> 7d GPU1
#Insects_128-007000_tmp2/ -> 8d GPU0
#Insects_128-007000_tmp3/ -> 8d GPU1
#Insects_128-007000_tmp4/ -> Titan GPU0

####################################################################################################
## for Birds:
## for step1: using orig Birds MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 126 \
    --data_path cGAN_data/generate/Birds_128-011200_tmp/ \
    --resume checkpoint/guo.1648/experiments/11528/checkpoint-45.pth

#12-152
#Birds_128-011200/ -> 7d GPU0
#Birds_128-011200_tmp/ -> 7d GPU1
#Birds_128-011200_tmp2/ -> 8d GPU0
#Birds_128-011200_tmp3/ -> 8d GPU1

####################################################################################################
## for Amphibians:
## for step1: using orig Amphibians MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 10 \
    --data_path cGAN_data/generate/Amphibians_128-009200/ \
    --resume checkpoint/guo.1648/experiments/11179/checkpoint-46.pth

#153-162
#Amphibians_128-009200/ -> Titan GPU0

####################################################################################################
## for Fungi:
## for step1: using orig Fungi MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 12 \
    --data_path cGAN_data/generate/Fungi_128-006800/ \
    --resume checkpoint/guo.1648/experiments/2568/checkpoint-18.pth

#0-11
#Amphibians_128-009200/ -> Titan GPU0


####################################################################################################
## for flowers:
## for step1: using orig flowers MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 5 \
    --data_path cGAN_data/generate/flowers_128-007600/ \
    --resume checkpoint/guo.1648/experiments/51956/checkpoint-29.pth

#0-4


####################################################################################################
## for UTKFace:
## for step1: using orig UTKFace MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 5 \
    --data_path cGAN_data/generate/UTKFace_128-004200/ \
    --resume checkpoint/guo.1648/experiments/21569/checkpoint-49.pth

#0-4


####################################################################################################
## for scene:
## for step1: using orig scene MAE classifider to select images: 
python imgSelectionMAE_chenqi.py \
    --eval  --model vit_huge_patch14 --batch_size 8 --nb_classes 6 \
    --data_path cGAN_data/generate/scene_128-004800/ \
    --resume checkpoint/guo.1648/experiments/39452/checkpoint-37.pth

#0-5















