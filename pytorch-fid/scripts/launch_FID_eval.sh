#!/bin/bash

# on each super-class:

# for iNaturalist dataset:

# FID:  57.669530841184326
python fid_score.py /eecf/cbcsl/data100b/Chenqi/pytorch-fid/data/aug-data/Amphibians \
 /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist_merge/train/Amphibians \
 --device cuda:0 --batch-size 50

# FID:  28.253868079416065
python fid_score.py /eecf/cbcsl/data100b/Chenqi/pytorch-fid/data/aug-data/Birds \
 /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist_merge/train/Birds \
 --device cuda:0 --batch-size 50

# FID:  50.0821023054616
python fid_score.py /eecf/cbcsl/data100b/Chenqi/pytorch-fid/data/aug-data/Fungi \
 /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/iNaturalist_merge/train/Fungi \
 --device cuda:0 --batch-size 50


# for scene dataset:

# FID:  54.3383226717267
python fid_score.py /eecf/cbcsl/data100b/Chenqi/pytorch-fid/data/aug-data/scene \
 /eecf/cbcsl/data100b/Chenqi/imbalanced_data/data/scene_merge/train \
 --device cuda:0 --batch-size 50


