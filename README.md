# SC_DA
## Abstract:
Unsupervised domain adaptation is critical in various computer vision tasks, such as object detection, instance segmentation, which aims to alleviate performance degradation caused by domain-shift and promotes the rapid application of models such as detection model in practical tasks. Previous works use adversarial training to align global features and local features to eventually minimize the domain discrepancy. However, there is no semantic information for each pix-feature in the local features, which causes the different categories of pixel-features to align and increase the difficulty of alignment. So a Semantic Consistency Alignment(SCA) model is proposed to align the pixel-feature with the same categories and mitigate negative transfer in this paper. Firstly, a Semantic Prediction Model(SPM) is designed in SCA to predict the categories of each pixel-feature, which is trained by the source domain according to the receptive field. Then the penultimate layer of SPM as a Semantic Vector Map(SVM) is jointed with the local feature to send to the local domain discriminator. Meanwhile, the output of SPM as an attention map is used to adjust the weight of the local domain discriminator. Extensive experiments for unsupervised domain adaptation on commonly used datasets demonstrate the effectiveness of our proposed approach for robust object detection in various domain shift scenarios.

## The models of our are:
https://pan.baidu.com/s/1LJgPuqJcCa4IjctwiPQxXQ      access code：opqh
 
## Refer to <Strong-Weak Distribution Alignment for Adaptive Object Detection>:
 https://github.com/VisionLearningGroup/DA_Detection

## Main requirements:
* torch == 1.0.0
* torchvision == 0.2.0
* Python 3
## Environmental settings：
This repository is developed using python 3.6.7 on Ubuntu 16.04.5 LTS. The CUDA nad CUDNN version is 9.0 and 7.4.1 respectively. We use one NVIDIA 1080ti GPU card for training and testing. Other platforms or GPU cards are not fully tested.

