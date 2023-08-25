# Foggy Cityscapes

*Foggy Cityscapes* derives from the Cityscapes dataset and constitutes a collection of partially synthetic foggy images that automatically inherit the semantic annotations of their real, clear counterparts in Cityscapes.

Details and **further downloads** for *Foggy Cityscapes* (including transmittance maps) are available at: 
people.ee.ethz.ch/~csakarid/SFSU_synthetic
For details on the original Cityscapes dataset, please refer to:
www.cityscapes-dataset.net


### Dataset Structure

The folder structure of *Foggy Cityscapes* follows that of Cityscapes, with minor extensions. Please refer to the README file of the Cityscapes git repository for a detailed presentation of this structure: 
https://github.com/mcordts/cityscapesScripts

In particular, following the notation in the aforementioned README, the extension field `ext` of the synthetic foggy versions of left 8-bit Cityscapes images includes additional information on the attenuation coefficient `beta` which was used to render them, as shown in the following sample image name:
```
erfurt_000000_000019_leftImg8bit_foggy_beta_0.01.png
```
Foggy images in the `train`, `val`, and `test` splits are available for `beta` equal to `0.005`, `0.01`, and `0.02`, while those in the `train_extra` split are available for `beta` equal to `0.01`.


### Foggy Cityscapes-refined

A refined list of 550 Cityscapes images (498 `train` plus 52 `val`) that yield high-quality synthetic foggy images is provided in the file `trainval_refined_filenames.txt`. Details on the refinement criteria are given in the relevant publication (see Citation below).


### Citation

If you use *Foggy Cityscapes* in your research, please cite both the Cityscapes publication and the publication that introduces *Foggy Cityscapes* as listed on the relevant website:
people.ee.ethz.ch/~csakarid/SFSU_synthetic


### Contact

Christos Sakaridis
csakarid@vision.ee.ethz.ch
people.ee.ethz.ch/~csakarid/SFSU_synthetic