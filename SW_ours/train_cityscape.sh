#!/bin/bash
save_dir="./reslut/experiments/SW_Faster_ICR_CCR/cityscape/ICR_CCR_23712_14_1"
dataset="cityscape"
pretrained_path="./pretrained/vgg16_caffe.pth"
net="vgg16"
#resume = True
#load_name = "./reslut/experiments/SW_Faster_ICR_CCR/cityscape/model/cityscape_2.pth"

#python da_train_net_SW.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex  --r True  --resume_name "cityscape_11.pth"
CUDA_VISIBLE_DEVICES=1 python da_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex
