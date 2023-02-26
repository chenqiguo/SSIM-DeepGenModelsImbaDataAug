#!/bin/bash

# on each super-class:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3/iNaturalist_eachSubCls/Insects \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3_iNaturalist/Insects

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/part3/checkpoint_Epoch70.pth.tar

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug2 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug2 \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug2/part4/checkpoint_Epoch80.pth.tar

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug3 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug3

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step2 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step2

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug3_step2 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug3_step2

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step3 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step3 \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step3/part4/checkpoint_Epoch80.pth.tar

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step1 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step1

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug3_step3 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug3_step3

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step2 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step2

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step4 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step4 \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step4/part4/checkpoint_Epoch80.pth.tar

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step3 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step3 \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step3/part2/checkpoint_Epoch40.pth.tar

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step4_v2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step4_v2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Insects_step4_v2/thresh_25 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Insects/step4_v2/thresh_25




python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step3_v2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step3_v2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step3_v2/thresh_30 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step3_v2/thresh_30

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step3_v2/thresh_40 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step3_v2/thresh_40


python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step4_v2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step4_v2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step4_v2/thresh_30 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step4_v2/thresh_30

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Fungi_forDebug_step4_v2/thresh_35 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Fungi/forDebug_step4_v2/thresh_35



python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_v3_cls/iNaturalist_eachSubCls/Amphibians_step1_v2/thresh_12 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_gan_v3cls_iNaturalist/Amphibians/step1_v2/thresh_12












python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_eachSubCls/Birds \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_gan_iNaturalist/Birds
 
python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_eachSubCls/Reptiles \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_gan_iNaturalist/Reptiles
  
python cls_res.py -a resnet18 --gpu 1 --epochs 100 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_eachSubCls/Amphibians \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cls_res18_gan_iNaturalist/Amphibians \
 
# on all sub-classes together:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 -b 128 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_allSubCls \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/cls_res18_gan_iNaturalist \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSubCls/cls_res18_gan_iNaturalist/part5/checkpoint_bestAcc1.pth.tar

# on all super-classes:

python cls_res.py -a resnet18 --gpu 0 --epochs 40 -b 128 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data/iNaturalist_allSuperCls \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSuperCls/cls_res18_gan_iNaturalist \
 --resume /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_allSuperCls/cls_res18_gan_iNaturalist/part1/checkpoint_bestAcc1.pth.tar


  
# to resume:
# --resume





