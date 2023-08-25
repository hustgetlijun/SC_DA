#!/bin/bash
save_dir="./data/experiments/SW_Faster_ICR_CCR/clipart/model_result/model_allover_all/clipart_1"
dataset="clipart"
pretrained_path="./pretrained/resnet101_caffe.pth"
net="res101"

CUDA_VISIBLE_DEVICES=1 python da_train_net_overall.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex
#--r Ture --resume_name clipart_8.pth