#!/bin/bash
save_dir="./data/experiments/SW_Faster_ICR_CCR/cityscape/model_result/model_pix_class_aera_0.6_one_cosin"
dataset="cityscape"
pretrained_path="./pretrained/vgg16_caffe.pth"
net="vgg16"

CUDA_VISIBLE_DEVICES=1 python da_train_net_pix_label.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex
#CUDA_VISIBLE_DEVICES=1 python da_train_net_pix_label.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex --r Ture --resume_name cityscape_8.pth