#!/bin/bash
#source ~/envs/ada/bin/activate; module load cuda/11.2.2 ninja

"""Generate images using pretrained network pickle.
Examples:

\b
# Generate curated MetFaces images without truncation (Fig.10 left)
python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

\b
# Generate uncurated MetFaces images with truncation (Fig.12 upper left)
python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

\b
# Generate class conditional CIFAR-10 images (Fig.17 left, Car)
python generate.py --outdir=out --seeds=0-35 --class=1 \\
--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

\b
# Render an image from projected W
python generate.py --outdir=out --projected_w=projected_w.npz \\
--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
"""



### our cmds:

## for Insects:

# seeds=600-3600, for a class 0-140 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Insects_128-007000/140 --trunc=1 --class=140 --seeds=600-3600 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=4600-7600, for b:
# cGAN->orig_cls:
# class 5->101, 69->102, 102->105, 75->106, 72->108, 85->109, 136->110, 37->111, 56->112, 83->113, 13->118, 113->119, 19->12, 21->130, 91->141, 97->143, 8->145, 131->146, 3->148, 117->15, 125->17, 39->61, 120->64,  108->66, 16->67, 130->72, 4->80, 57->82, 133->83, 35->84, 45->87, 47->89, 95->94 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Insects_128-007000/b/94 --trunc=1 --class=95 --seeds=4600-7600 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=8600-11600, for c:
# cGAN->orig_cls:
# class 5->101, 102->105, 75->106, 37->111, 8->145, 3->148, 108->66, 16->67, 4->80, 133->83, 35->84, 45->87 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Insects_128-007000/c/87 --trunc=1 --class=45 --seeds=8600-11600 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=11700-25000, for d:
# cGAN->orig_cls:
# class 19->12, 24->14, 117->15, 62->16, 125->17, 137->18, 126->19, 88->20, 103->27, 66->32, 135->40, 115->41, 101->42, 89->44, 96->47, 68->48, 105->49, 119->50, 74->53, 76->55, 67->58, 139->59, 30->60, 39->61, 11->62, 44->63, 120->64, 65->65, 108->66, 16->67, 40->71, 130->72, 99->73, 63->74, 93->75, 138->76, 29->79, 4->80, 133->83, 122->85, 45->87, 43->90, 64->92, 95->94, 86->99, 26->103, 102->105, 75->106, 72->108, 136->110, 37->111, 56->112, 83->113, 60->115, 106->116, 111->117, 13->118, 113->119, 118->120, 38->121, 32->122, 25->123, 46->125, 7->126, 124->128, 49->129, 21->130, 12->131, 14->132, 70->133, 110->134, 100->135, 129->136, 17->137, 50->138, 121->139, 91->141, 23->142, 97->143, 8->145, 131->146, 3->148, 123->149, 134->150, 1->151 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Insects_128-007000/d/151 --trunc=1 --class=1 --seeds=11700-25000 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=25100-55100, for e:
# cGAN->orig_cls:
# class 19->12, 117->15, 125->17, 137->18, 126->19, 88->20, 115->41, 101->42, 89->44, 105->49, 76->55, 139->59, 30->60, 39->61, 11->62, 65->65, 108->66, 16->67, 138->76, 4->80, 133->83, 45->87, 86->99, 75->106, 72->108, 136->110, 37->111, 83->113, 60->115, 111->117, 13->118, 113->119, 32->122, 124->128, 49->129, 21->130, 14->132, 70->133, 110->134, 17->137, 97->143, 131->146, 3->148, 1->151 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Insects_128-007000/e/151 --trunc=1 --class=1 --seeds=25100-55100 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=55200-355200 (at most), for f:
# cGAN->orig_cls:
# for seeds=55200-65200: class 11->62, 16->67, 14->132, 97->143, :
# for seeds=55200-85200: class 137->18, 89->44, 45->87, 70->133, :
# for seeds=55200-105200: class 76->55, 133->83, :
# for seeds=55200-115200: class 75->106, 37->111, 60->115, :
# for seeds=55200-125200: class 32->122, :
# for seeds=55200-155200: class 111->117, :
# for seeds=55200-355200: class 30->60, 39->61, 108->66, 72->108, 136->110, 83->113, 124->128, 21->130, 110->134, 17->137 :
python generate.py --outdir=results/generate/Insects_128-007000/f/137 --trunc=1 --class=17 --seeds=55200-355200 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl


# seeds=355300-955300 (at most), for g:
# cGAN->orig_cls:
# for seeds=355300-405300: class 133->83, :
# for seeds=355300-415300: class 75->106, :
# for seeds=355300-455300: class 32->122, :
# for seeds=355300-955300: class 39->61, 72->108, 83->113, 21->130, 17->137 :
python generate.py --outdir=results/generate/Insects_128-007000/g/137 --trunc=1 --class=17 --seeds=355300-955300 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=955400-1955400, for h:
# cGAN->orig_cls:
# for seeds=955400-1955400: class 72->108, 83->113, 17->137 :
python generate.py --outdir=results/generate/Insects_128-007000/h/137 --trunc=1 --class=17 --seeds=955400-1955400 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=1955500-2955500, for i:
# cGAN->orig_cls:
# for seeds=1955500-2955500: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/i/113 --trunc=1 --class=83 --seeds=1955500-2955500 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=2955600-3955600, for j:
# cGAN->orig_cls:
# for seeds=2955600-3955600: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/j/113 --trunc=1 --class=83 --seeds=2955600-3955600 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=3955700-4955700, for k:
# cGAN->orig_cls:
# for seeds=3955700-4955700: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/k/113 --trunc=1 --class=83 --seeds=3955700-4955700 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=4955800-5955800, for l:
# cGAN->orig_cls:
# for seeds=4955800-5955800: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/l/113 --trunc=1 --class=83 --seeds=4955800-5955800 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=5955900-6955900, for m:
# cGAN->orig_cls:
# for seeds=5955900-6955900: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/m/113 --trunc=1 --class=83 --seeds=5955900-6955900 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=6955910-7955910, for n:
# cGAN->orig_cls:
# for seeds=6955910-7955910: class 72->108 :
python generate.py --outdir=results/generate/Insects_128-007000/n/108 --trunc=1 --class=72 --seeds=6955910-7955910 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=7955920-8955920, for o:
# cGAN->orig_cls:
# for seeds=7955920-8955920: class 72->108 :
python generate.py --outdir=results/generate/Insects_128-007000/o/108 --trunc=1 --class=72 --seeds=7955920-8955920 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=8955930-9955930 (at most), for p:
# cGAN->orig_cls:
# for seeds=8955930-8985930: class 19->12, 137->18, 105->49, 30->60, 108->66, 16->67, 112->68, 4->80, 45->87, 64->92, 55->93, 2->95, 85->109, 37->111, 27->114, 46->125, 49->129, 14->132, 17->137, 121->139, 107->140, 97->143, 8->145, 131->146, 3->148, :
# for seeds=8955930-9955930: class 133->83, 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/p/148 --trunc=1 --class=3 --seeds=8955930-8985930 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=9955940-10955940 (at most), for q:
# cGAN->orig_cls:
# for seeds=9955940-9985940: class 30->60, 133->83, :
# for seeds=9955940-10015940: class 108->66, :
# for seeds=9955940-10955940: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/q/113 --trunc=1 --class=83 --seeds=9955940-10955940 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=10955950-11955950 (at most), for r:
# cGAN->orig_cls:
# for seeds=10955950-10985950: class 133->83, :
# for seeds=10955950-11015950: class 108->66, :
# for seeds=10955950-11955950: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/r/113 --trunc=1 --class=83 --seeds=10955950-11955950 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=11955960-12955960 (at most), for s:
# cGAN->orig_cls:
# for seeds=11955960-11985960: class 133->83, :
# for seeds=11955960-12955960: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/s/113 --trunc=1 --class=83 --seeds=11955960-12955960 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

# seeds=12955970-14955970 (at most), for t:
# cGAN->orig_cls:
# for seeds=12955970-13015970: class 133->83, :
# for seeds=12955970-14955970: class 72->108, 83->113 :
python generate.py --outdir=results/generate/Insects_128-007000/t/113 --trunc=1 --class=83 --seeds=12955970-14955970 \
--network=results/training-runs/00002-Insects_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-007000.pkl

####################################################################################################

## for Fungi step1:
# seeds=700-5700, a, for class 0-11 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Fungi_128-006800/11 --trunc=1 --class=11 --seeds=700-5700 \
--network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl

# seeds=6700-16700, for b:
# cGAN->orig_cls:
# class 11->0, 8->2, 1->7, 3->8, 0->9, 2->10, 6->11, :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Fungi_128-006800/b/11 --trunc=1 --class=6 --seeds=6700-16700 \
--network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl

# seeds=16800-1016800 (at most), for c:
# cGAN->orig_cls:
# for seeds=16800-26800: class 3->8, :
# for seeds=16800-1016800: class 11->0, 1->7, 2->10, :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Fungi_128-006800/c/10 --trunc=1 --class=2 --seeds=16800-1016800 \
--network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl

# seeds=1026820-2026820, for d:
# cGAN->orig_cls: class 2->10, :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Fungi_128-006800/d/10 --trunc=1 --class=2 --seeds=1026820-2026820 \
--network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl


## for Fungi step2:
# seeds=1016810-1026810, for b:
# cGAN->orig_cls: class 10->4, 7->6, 0->9, 6->11, :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Fungi_128-006800_step2/b/11 --trunc=1 --class=6 --seeds=1016810-1026810 \
--network=results/training-runs/00001-Fungi_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-006800.pkl

####################################################################################################

## for Birds step1:
# seeds=700-5700, a, for class 0-125 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200/a/125 --trunc=1 --class=125 --seeds=700-5700 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=5800-15800, b, for class:
# cGAN->orig_cls: 81->202, 94->204, 64->205, 36->206, 70->207, 24->208, 66->210, 61->212, 92->213, 115->214, 25->215, 111->216, 107->217, 15->218, 37->221, 33->222, 20->223, 89->224, 5->227, 11->228, 123->229, 84->230, 16->233, 28->234, 83->235, 41->238, 51->239, 122->240, 34->241, 104->242, 29->243, 124->245, 114->248, 95->253, 40->255, 98->256, 21->257, 103->258, 44->259, 79->260, 116->261, 12->262, 0->263, 86->264, 80->265, 113->269, 117->271, 97->272, 88->273, 6->274, 30->276, 52->277, 106->279, 62->280, 68->283, 35->286, 100->287, 46->289, 108->290, 91->292, 23->293, 76->294, 69->295, 59->296, 93->297, 56->298, 109->300, 10->302, 22->303, 13->304, 26->305, 58->307, 75->309, 85->310, 54->312, 96->315, 110->316, 57->317, 47->318, 32->321, 78->322, 73->324, 102->326, 49->327 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200/b/327 --trunc=1 --class=49 --seeds=5800-15800 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=15900-45900, c, for class:
# cGAN->orig_cls: 36->206, 70->207, 24->208, 66->210, 115->214, 25->215, 111->216, 20->223, 89->224, 5->227, 11->228, 123->229, 84->230, 83->235, 51->239, 34->241, 104->242, 29->243, 124->245, 98->256, 21->257, 103->258, 116->261, 86->264, 113->269, 117->271, 97->272, 6->274, 35->286, 100->287, 46->289, 23->293, 56->298, 109->300, 10->302, 22->303, 75->309, 96->315, 57->317 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200/c/317 --trunc=1 --class=57 --seeds=15900-45900 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=46000-106000, d, for class:
# cGAN->orig_cls: 25->215, 111->216, 11->228, 29->243, 86->264, 46->289, 10->302, 96->315 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200/d/315 --trunc=1 --class=96 --seeds=46000-106000 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=106001-136001, e, for class:
# cGAN->orig_cls: 38->211, 20->223, 34->241, 112->249, 40->255, 44->259, 79->260, 7->267, 45->270, 108->290, 53->299, 4->301, 10->302, 96->315, 57->317, 47->318, 63->319, 67->325, 49->327 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200/e/327 --trunc=1 --class=49 --seeds=106001-136001 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl


## for Birds step2:
# seeds=136002-166002, b, for class:
# cGAN->orig_cls: 37->221, 42->266, 14->275, 106->279, 101->281 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200_step2/b/281 --trunc=1 --class=101 --seeds=136002-166002 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=166003-196003, c, for class:
# cGAN->orig_cls: 33->222, 27->225, 19->237, 51->239, 122->240, 29->243, 99->244, 17->254, 18->288, 26->305, 85->310, 73->324 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200_step2/c/324 --trunc=1 --class=73 --seeds=166003-196003 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=196004-226004, d, for class:
# cGAN->orig_cls: 81->202, 3->203, 36->206, 24->208, 74->209, 61->212, 92->213, 115->214, 25->215, 111->216, 107->217, 15->218, 71->219, 72->220, 33->222, 20->223, 89->224, 2->226, 5->227, 123->229, 84->230, 120->232, 28->234, 83->235, 119->236, 41->238, 51->239, 122->240, 34->241, 104->242, 29->243, 124->245, 50->246, 114->248, 118->250, 87->251, 95->253, 40->255, 21->257, 103->258, 116->261, 12->262, 0->263, 86->264, 80->265, 105->268, 117->271, 88->273, 30->276, 125->278, 62->280, 82->282, 68->283, 77->284, 48->285, 35->286, 46->289, 108->290, 121->291, 91->292, 76->294, 69->295, 93->297, 56->298, 109->300, 10->302, 13->304, 58->307, 54->312, 55->313, 110->316, 57->317, 47->318, 9->320, 32->321, 65->323, 73->324, 102->326, 49->327 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200_step2/d/327 --trunc=1 --class=49 --seeds=196004-226004 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=226005-326005 (at most), e, for class: cGAN->orig_cls: 
# for seeds=226005-256005: 25->215, 104->242, 116->261, 102->326 :
# for seeds=226005-326005: 29->243 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200_step2/e/326 --trunc=1 --class=102 --seeds=226005-256005 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=326006-426006, f, for class:
# cGAN->orig_cls: 29->243 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Birds_128-011200_step2/f/243 --trunc=1 --class=29 --seeds=326006-426006 \
--network=results/training-runs/00005-Birds_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl



####################################################################################################

## for Reptiles step1:
# seeds=700-5700, a, for class 0-38 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200/a/38 --trunc=1 --class=38 --seeds=700-5700 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=5800-35800, b, for class:
# cGAN->orig_cls: 37->163, 18->164, 4->165, 1->166, 5->167, 38->168, 24->169, 22->170, 10->171, 20->173, 21->174, 34->175, 15->176, 12->177, 0->178, 25->179, 35->180, 19->181, 31->182, 8->183, 3->184, 7->185, 13->186, 33->188, 36->189, 23->190, 6->191, 14->192, 32->193, 16->194, 29->195, 27->196, 11->197, 26->198, 28->199, 17->200, 30->201 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200/b/201 --trunc=1 --class=30 --seeds=5800-35800 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=35900-135900, c, for class:
# cGAN->orig_cls: 4->165, 1->166, 10->171, 20->173, 15->176, 25->179, 31->182, 3->184, 28->199 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200/c/199 --trunc=1 --class=28 --seeds=35900-135900 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=136000-166000, d, for class:
# cGAN->orig_cls: 25->179 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200/d/179 --trunc=1 --class=25 --seeds=136000-166000 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=166001-266001, e, for class:
# cGAN->orig_cls: 4->165, 22->170, 2->172, 25->179, 8->183, 3->184, 13->186, 9->187, 11->197 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200/e/197 --trunc=1 --class=11 --seeds=166001-266001 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=266002-366002 (at most), f, for class:
# cGAN->orig_cls: 4->165, 25->179, 8->183 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200/f/179 --trunc=1 --class=25 --seeds=266002-366002 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl


## for Reptiles step2:
# seeds=366003-566003, b, for class:
# cGAN->orig_cls: 25->179:
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200_step2/b/179 --trunc=1 --class=25 --seeds=366003-566003 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl

# seeds=566004-586004, c, for class:
# cGAN->orig_cls: 4->165, 2->172, 9->187, 11->197 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Reptiles_128-011200_step2/c/197 --trunc=1 --class=11 --seeds=566004-586004 \
--network=results/training-runs/00006-Reptiles_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-011200.pkl


####################################################################################################

## for Amphibians step1:
# seeds=700-10700, a, for class 0-9 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Amphibians_128-009200/a/9 --trunc=1 --class=9 --seeds=700-10700 \
--network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl

# seeds=110702-210702, a, for class: 8->154, 6->155, 7->157, 1->162, :
# seeds=110702-140702, a, for class: 5->158, 4->160 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Amphibians_128-009200/a/160 --trunc=1 --class=4 --seeds=110702-140702 \
--network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl

# seeds=210703-310703, b, for class: 3->159, 0->161, :
# seeds=210703-240703, b, for class: 2->156 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Amphibians_128-009200/b/161 --trunc=1 --class=0 --seeds=210703-310703 \
--network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl

# seeds=310704-510704, c, for class: 8->154, 6->155 :
CUDA_VISIBLE_DEVICES=1 python generate.py --outdir=results/generate/Amphibians_128-009200/c/155 --trunc=1 --class=6 --seeds=310704-510704 \
--network=results/training-runs/00007-Amphibians_128-cond-mirror-trans:2000-4000-auto1/network-snapshot-009200.pkl
































