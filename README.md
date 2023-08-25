# SC_DA
## Abstract:
Unsupervised domain adaptation is crucial for mitigating the performance degradation caused by domain bias in object detection tasks. Previous studies focus on pix-level and instance-level shifts alignment in attempts to minimize domain discrepancy. However, this method may lead to aligning single-class features with mixed-class features during image-level domain adaptation, given that each image in object detection tasks can belong to more than one category. To achieve the same category feature alignment between single-class and mixed-class, our method considers features with different mixed categories as a new class and proposes a mixed-classes $H-divergence$ to reduce domain bias for object detection. To enhance both single-class and mixed-class semantic information and achieve semantic separation for mixed-classes $H-divergence$, we employ Semantic Prediction Models (SPM) and Semantic Bridging Components (SBC). Furthermore, we reweigh the loss of the pix domain discriminator based on the SPM results to reduce sample imbalance. Our extensive experiments on widely used datasets illustrate how our method can robustly improve object detection in domain bias settings.

## The Datasets of ours are:
https://pan.baidu.com/s/1ZiEdHgRVhmBZvywUhvbBkQ              access code：odq6 

## pre-trained models:
https://pan.baidu.com/s/1yuwSkMjKSFr0InZAsHTs7Q             access code: qhr8
 
## Related works:
 * https://github.com/hustgetlijun/RCAN

## Main requirements:
* torch == 1.0.0
* torchvision == 0.2.0
* Python 3

# To train SW-ours on cityscape:
sh train_cityscape_allover.sh
# To validate SW-ours on cityscape:
python test_cityscape_change.py

## Environmental settings：
This repository is developed using python 3.6.7 on Ubuntu 16.04.5 LTS. The CUDA nad CUDNN version is 10.0 and 7.4.1 respectively. We use one NVIDIA 2080ti GPU card for training and testing. Other platforms or GPU cards such as V100 are tested.
 
## Citing this repository：
 

