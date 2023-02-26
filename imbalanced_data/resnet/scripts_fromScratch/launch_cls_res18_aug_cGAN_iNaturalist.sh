#!/bin/bash

## on each super-class:

# for Insects:

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Insects/opt1 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Insects/opt1

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Insects/opt2/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Insects/opt2/step1/thresh_10

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Insects/opt2/step2/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Insects/opt2/step2/thresh_10

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Insects/opt2/step2/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Insects/opt2/step2/thresh_15



python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Insects \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/Insects



# for Fungi:

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Fungi/opt2/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step1/thresh_10

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Fungi/opt2/step1/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step1/thresh_15



python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Fungi/opt2/step2/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step2/thresh_10

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Fungi/opt2/step2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Fungi/opt2/step2/thresh_30 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Fungi/opt2/step2/thresh_30


python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Fungi \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/Fungi


# for Birds:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Birds/opt2/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step1/thresh_10

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Birds/opt2/step1/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step1/thresh_15


python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Birds/opt2/step2/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step2/thresh_15

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Birds/opt2/step2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Birds/opt2/step2/thresh_25 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Birds/opt2/step2/thresh_25


python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Birds \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/Birds


# for Reptiles:

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Reptiles/opt2/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Reptiles/opt2/step1/thresh_10

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Reptiles/opt2/step1/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Reptiles/opt2/step1/thresh_15


python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Reptiles/opt2/step2/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Reptiles/opt2/step2/thresh_15

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Reptiles/opt2/step2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Reptiles/opt2/step2/thresh_20


python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Reptiles \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/Reptiles

# for Amphibians:

python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Amphibians/opt2/step1/thresh_10 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Amphibians/opt2/step1/thresh_10

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Amphibians/opt2/step1/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Amphibians/opt2/step1/thresh_15

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Amphibians/opt2/step2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Amphibians/opt2/step2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_eachSubCls/Amphibians/opt2/step2/thresh_30 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_eachSubCls/cls_res18_cGAN_iNaturalist/Amphibians/opt2/step2/thresh_30


python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Amphibians \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/Amphibians

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/Amphibians_2ndTry \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/Amphibians_2ndTry



## for cGAN-aug iNatruarlist (all except Plants):
python cls_res.py -a resnet18 --gpu 0 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/iNaturalist_cGANaug \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/iNaturalist_cGANaug_allButPlants





# for flowers:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/flowers/opt2/step1/thresh_15 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/flowers/cls_res18_cGAN/opt2/step1/thresh_15

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/flowers/opt2/step2/thresh_20 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/flowers/cls_res18_cGAN/opt2/step2/thresh_20

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/flowers \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/flowers


# for UTKFace:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/UTKFace/opt2/step1/thresh_25 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/UTKFace/cls_res18_cGAN/opt2/step1/thresh_25

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/UTKFace/opt2/step2/thresh_25 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/UTKFace/cls_res18_cGAN/opt2/step2/thresh_25


python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/UTKFace \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/UTKFace


# for scene:

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/scene/opt2/step1/thresh_30 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/cls_res18_cGAN/opt2/step1/thresh_30

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/scene/opt2/step2/thresh_30 \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/scene/cls_res18_cGAN/opt2/step2/thresh_30

python cls_res.py -a resnet18 --gpu 1 --epochs 100 --saveEveryEpoch 5 \
 --data /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/aug_data_cGAN/without_cls_select/scene \
 --result /eecf/cbcsl/data100b/Chenqi/imbalanced_data/resnet/results_fromScratch/cGANaug_without_cls_select/scene


  
# to resume:
# --resume





