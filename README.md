# SC_DA
## Abstract:
Unsupervised domain adaptation is critical in various computer vision tasks, such as object detection, instance segmentation, etc. They attempt to reduce domain bias-induced performance degradation while also promoting model application speed. Previous works in domain adaptation object detection attempt to align image-level and instance-level shifts to eventually minimize the domain discrepancy, but they may align single-class features to mixed-class features in image-level domain adaptation because each image in the object detection task may be more than one class and object. In order to achieve single-class with single-class alignment and mixed-class with mixed-class alignment, we treat the mixed-class of the feature as a new class and propose a mixed-classes $H-divergence$ for object detection to achieve homogenous feature alignment and reduce negative transfer. Then, a Semantic Consistency Feature Alignment Model (SCFAM) based on mixed-classes $H-divergence$ was also presented. To improve single-class and mixed-class semantic information and accomplish semantic separation, the SCFAM model proposes Semantic Prediction Models (SPM) and Semantic Bridging Components (SBC). And the weight of the pix domain discriminator loss is then changed based on the SPM result to reduce sample imbalance. Extensive unsupervised domain adaption experiments on widely used datasets illustrate our proposed approach's robust object detection in domain bias settings.

## The Datasets of our are:
https://pan.baidu.com/s/1ZiEdHgRVhmBZvywUhvbBkQ      access code：odq6 
 
## Related works:
 * https://github.com/hustgetlijun/RCAN

## Main requirements:
* torch == 1.0.0
* torchvision == 0.2.0
* Python 3
## Environmental settings：
This repository is developed using python 3.6.7 on Ubuntu 16.04.5 LTS. The CUDA nad CUDNN version is 9.0 and 7.4.1 respectively. We use one NVIDIA 1080ti GPU card for training and testing. Other platforms or GPU cards are not fully tested.
 
## Citing this repository：
 

