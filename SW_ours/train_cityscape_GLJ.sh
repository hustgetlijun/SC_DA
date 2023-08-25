#!/bin/bash
save_dir="./reslut/experiments/SW_Faster_ICR_CCR/cityscape/model_lc"
dataset="cityscape"
pretrained_path="./pretrained/vgg16_caffe.pth"
net="vgg16"
#resume = True
#load_name = "./reslut/experiments/SW_Faster_ICR_CCR/cityscape/model/cityscape_2.pth"

CUDA_VISIBLE_DEVICES=0 python da_train_GLJ.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex  --r True  --resume_name "cityscape_3.pth"
#python da_train_GLJ.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex