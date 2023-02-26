

### Script from MD: a baseline training:
# (1) sampling from the diffusion model (base): --> NOT use!!!

# (2) sampling from the classifier model:
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS



### An example for debug: sampling with pretrained ImageNet 128x128 model:
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
python scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path pretrained_models/ImageNet/128x128_classifier.pt --model_path pretrained_models/ImageNet/128x128_diffusion.pt $SAMPLE_FLAGS



### our cmd:

####################################################################################################
## for Fungi:
# (2) sampling from the classifier model (from scratch): --> NOT use!!!
python scripts/classifier_sample.py --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 4000 --image_size 128 --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
 --classifier_scale 0.5 --classifier_path results/classifier_diffusion/Fungi/classifier/model280000.pt \
 --model_path results/classifier_diffusion/Fungi/diffusion_model/ema_0.9999_100000.pt \
 --batch_size 20 --num_samples 100 --timestep_respacing 250

# (2.5) sampling from the classifier model ( w. diffu-model resumed on ImageNet pretrained model): --> use this!!!
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 70 --num_samples 40000 --timestep_respacing 250"
python scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path results/classifier_diffusion/Fungi/classifier/model280000.pt --model_path results/classifier_diffusion/Fungi/diffusion_model_resumed/ema_0.9999_090000.pt $SAMPLE_FLAGS


####################################################################################################
## for Birds:
# (2.5) sampling from the classifier model ( w. diffu-model resumed on ImageNet pretrained model): --> use this!!!
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 70 --num_samples 40000 --timestep_respacing 250"
CUDA_VISIBLE_DEVICES=1 python scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path results/classifier_diffusion/Birds/classifier/model060000.pt --model_path results/classifier_diffusion/Birds/diffusion_model_resumed/ema_0.9999_110000.pt $SAMPLE_FLAGS


####################################################################################################
## for scene:
# (2.5) sampling from the classifier model ( w. diffu-model resumed on ImageNet pretrained model): --> use this!!!
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 70 --num_samples 4000 --timestep_respacing 250"
python scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path results/classifier_diffusion/scene/classifier/model080000.pt --model_path results/classifier_diffusion/scene/diffusion_model_resumed/ema_0.9999_100000.pt $SAMPLE_FLAGS


####################################################################################################
## for Amphibians:
# (2.5) sampling from the classifier model ( w. diffu-model resumed on ImageNet pretrained model): --> use this!!!
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 70 --num_samples 10000 --timestep_respacing 250"
CUDA_VISIBLE_DEVICES=1 python scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path results/classifier_diffusion/Amphibians/classifier/model100000.pt --model_path results/classifier_diffusion/Amphibians/diffusion_model_resumed/ema_0.9999_100000.pt $SAMPLE_FLAGS









