

### Script from MD: a baseline training:
# (1) training the diffusion model:
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# (2) training the classifier model:
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
python scripts/classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS



### our cmd:

####################################################################################################
## for Fungi:
# (1) training the diffusion model (from scratch): --> NOT use!!!
python scripts/image_train.py --data_dir /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/Fungi \
 --image_size 128 --num_channels 128 --num_res_blocks 3 \
 --diffusion_steps 4000 --noise_schedule linear \
 --lr 1e-4 --batch_size 20

# (1.5) training the diffusion model (resumed on ImageNet pretrained model): --> first run this!!!
MODEL_FLAGS="--image_size 128 --num_channels 256 --num_res_blocks 2 --resblock_updown True --class_cond True --learn_sigma True --num_channels 256 --num_heads 4 --attention_resolutions 32,16,8 --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10 --use_fp16 True --timestep_respacing 250"
python scripts/image_train.py --data_dir datasets/my_data/Fungi \
 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
 --resume_checkpoint pretrained_models/ImageNet/128x128_diffusion.pt

# (2) training the classifier model: --> then run this!!!
python scripts/classifier_train.py --data_dir /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/Fungi \
 --iterations 300000 --anneal_lr True --batch_size 56 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 \
 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True


####################################################################################################
## for Birds:
# (1.5) training the diffusion model (resumed on ImageNet pretrained model): --> first run this!!! (run on 7d machine)
MODEL_FLAGS="--image_size 128 --num_channels 256 --num_res_blocks 2 --resblock_updown True --class_cond True --learn_sigma True --num_channels 256 --num_heads 4 --attention_resolutions 32,16,8 --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10 --use_fp16 True --timestep_respacing 250"
python scripts/image_train.py --data_dir datasets/my_data/Birds \
 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
 --resume_checkpoint pretrained_models/ImageNet/128x128_diffusion.pt

# (2) training the classifier model: --> then run this!!! (run on 8d machine)
python scripts/classifier_train.py --data_dir /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/Birds \
 --val_data_dir /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist/val/Birds \
 --iterations 300000 --anneal_lr True --batch_size 56 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 \
 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True


####################################################################################################
## for scene:
# (1.5) training the diffusion model (resumed on ImageNet pretrained model): --> first run this!!! (run on 8d machine GPU0)
MODEL_FLAGS="--image_size 128 --num_channels 256 --num_res_blocks 2 --resblock_updown True --class_cond True --learn_sigma True --num_channels 256 --num_heads 4 --attention_resolutions 32,16,8 --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10 --use_fp16 True --timestep_respacing 250"
python scripts/image_train.py --data_dir datasets/my_data/scene \
 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
 --resume_checkpoint pretrained_models/ImageNet/128x128_diffusion.pt

# (2) training the classifier model: --> then run this!!! (run on 8d machine GPU1)
python scripts/classifier_train.py --data_dir /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/scene \
 --val_data_dir /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene/scene/val \
 --iterations 300000 --anneal_lr True --batch_size 56 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 \
 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True


####################################################################################################
## for Amphibians:
# (1.5) training the diffusion model (resumed on ImageNet pretrained model): --> first run this!!! (run on 8d machine GPU0)
MODEL_FLAGS="--image_size 128 --num_channels 256 --num_res_blocks 2 --resblock_updown True --class_cond True --learn_sigma True --num_channels 256 --num_heads 4 --attention_resolutions 32,16,8 --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10 --use_fp16 True --timestep_respacing 250"
python scripts/image_train.py --data_dir datasets/my_data/Amphibians \
 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
 --resume_checkpoint pretrained_models/ImageNet/128x128_diffusion.pt

# (2) training the classifier model: --> then run this!!! (run on 8d machine GPU1)
python scripts/classifier_train.py --data_dir /eecf/cbcsl/data100b/Chenqi/guided-diffusion/datasets/my_data/Amphibians \
 --iterations 300000 --anneal_lr True --batch_size 56 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 \
 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True


#--val_data_dir /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist/val/Amphibians \









