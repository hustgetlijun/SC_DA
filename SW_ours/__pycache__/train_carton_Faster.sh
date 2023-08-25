#!/bin/bash
#save_dir="./data/experiment/Faster_ori/carton/train_G_LFM_CPLC_pre"
#dataset="carton"
#pretrained_path="./pretrained/VGG16_LFM_pre_ad.pth"
#net="vgg16"

save_dir="./data/experiment/Faster_ori/carton_four_label/train_G_ECLC"
dataset="carton"
pretrained_path="./pretrained/vgg16_caffe.pth"
net="vgg16"

CUDA_VISIBLE_DEVICES=2 python faster_train_net.py --max_epochs 6 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex
#CUDA_VISIBLE_DEVICES=1 python da_train_net_pix_label.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex --r Ture --resume_name cityscape_8.pth

#model_allover_128c_two
#model_allover_128c_tree_struct