#!/bin/bash

tar -xvf datasets/ImageNet_Carnivores_20_100.tar -C ./datasets

python dataset_tool.py --source=datasets/ImageNet_Carnivores_20_100 --dest=datasets/ImageNet_Carnivores_20_100.zip --transform=center-crop --width=128 --height=128


### our cmds:

python dataset_tool.py --source=datasets/Fungi --dest=datasets/Fungi_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/Insects --dest=datasets/Insects_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/Birds --dest=datasets/Birds_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/Reptiles --dest=datasets/Reptiles_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/Amphibians --dest=datasets/Amphibians_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/flowers --dest=datasets/flowers_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/UTKFace --dest=datasets/UTKFace_128.zip --transform=center-crop --width=128 --height=128

python dataset_tool.py --source=datasets/scene --dest=datasets/scene_128.zip --transform=center-crop --width=128 --height=128








