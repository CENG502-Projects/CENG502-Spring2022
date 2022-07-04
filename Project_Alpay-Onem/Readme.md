# Ghost Removal via Channel Attention in Exposure Fusion

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

PyTorch implementation of "Ghost Removal via Channel Attention in Exposure Fusion" (MCANet) [1].

*Yan, Qingsen, Bo Wang, Peipei Li, Xianjun Li, Ao Zhang, Qinfeng Shi, Zheng You, Yu Zhu, Jinqiu Sun, and Yanning Zhang. "Ghost removal via channel attention in exposure fusion." Computer Vision and Image Understanding 201 (2020): 103079.*

The aim of this repository is to reproduce the results of the paper by implementing the code from scratch in PyTorch, to train on the same dataset by using the details described in the paper in exact (and assume the unmentioned details), and to try to obtain the same PSNR-mu (tonemapped by mu-law tonemapper) and PSNR-L (linear HDR domain) metric results on the test set as the paper. Some of this text includes extracts taken directly from the paper (text and figures).

## 1.1. Paper summary

The paper's abstract provides a good summary of the HDR deghosting field, together with what the paper's novelties are:

*"High dynamic range (HDR) imaging is to reconstruct high-quality images with a broad range of illuminations
from a set of differently exposed images. Some existing algorithms align the input images before merging them
into an HDR image, but artifacts of the registration appear due to misalignment. Recent works try to remove
the ghosts by detecting motion region or skipping the registered process, however, the result still suffers from
ghost artifacts for scenes with significant motions. In this paper, we propose a novel Multi-scale Channel
Attention guided Network (MCANet) to address the ghosting problem. We use multi-scale blocks consisting
of dilated convolution layers to extract informative features. The channel attention blocks suppress undesired
components and guide the network to refine features to make full use of feature maps. The proposed MCANet
recovers the occluded or saturated details and reduces artifacts due to misalignment. Experiments show that
the proposed MCANet can achieve state-of-the-art quantitative and qualitative results."*

# 2. The method and reproduction details

## 2.1. The original method

- Input: 3 images with different exposures, middle image is taken as reference.
- Output: Ghost-free HDR image, aligned to the reference image.
- Network: Composed of 2 modules, **feature extraction** and **reconstruction**. Paper focuses on the feature extraction part.
- Contributions: Mainly on the feature extraction part:
  - **M**ultiscale **C**hannel **A**ttention **B**lock (**MCAB**)
    - Helps to extract features from LDR images more effectively.
  - **C**hannel **A**ttention **B**lock (**CAB**)
    - Chooses the informative details and suppress undesired components caused by saturation or other artifacts.
  - Dilated convolutions provide better features with wider receptive range.
  - Channel-wise attention seems to perform better than spatial attention, since the quality of the non-reference image features do not spatially vary too much.


![image](https://user-images.githubusercontent.com/7654135/177110248-1050c7fc-075b-4227-8001-1359b04245b8.png)
<p align="center">
The flow of the method.
</p>


![image](https://user-images.githubusercontent.com/7654135/177110316-eb26fe94-b5f5-4081-bd49-1a6c0c52e051.png)
<p align="center">
MCAB structure.
</p>


![image](https://user-images.githubusercontent.com/7654135/177110391-2efb48ba-754c-4478-a0a6-1721b789d7bd.png)
<p align="center">
CAB structure.
</p>

- First, degamma is applied with *gamma=2.2* to the input images to put them in linear domain. They are then exposure-normalized and brought to the HDR domain. Both LDR and HDR versions are supplied to network (input channel size is 3x3x2 = 18, whereas the ground-truth HDR image has 3 channels). The network output has 3 channels.

- Differentiable tonemapping operation is applied to both the network output and the ground-truth HDR:

<p align="center">
  <img src="https://user-images.githubusercontent.com/7654135/177111577-8928b27d-8894-4072-a700-b9b763cae736.png">
</p>

- Trained on L1 loss between tonemapped output and ground-truth, using Adam optimizer.

<p align="center">
  <img src="https://user-images.githubusercontent.com/7654135/177111666-14c24545-8065-4599-9565-c3351140f53a.png">
</p>


## 2.2. Reproduction Issues 

- Training is made with 256x256 patches, however paper does not specify how test images are treated (patches/whole image).
  - We assumed that the test images were not divided into patches for the network. We process them as a whole. This caused high amount of GPU mem usage. Therefore, we have used the Grid DL Infrastructure's A100 GPU with 80 gigs of GPU mem for the training of the network + the test phase.
- Details regarding data augmentation missing.
  - We have applied horizontal/vertical flips, rotation, random (256x256) crops during training phase.
- The way output values are constrained to [0-255] or [0-1] is not specified.
  - We added a sigmoid layer at the end of the network. 
- No validation protocol specified.
- Information regarding training iterations is missing.
  - We terminated training after 500 epochs, where the loss curve seemed to converge.
- Discussion on Reconstruction Network is not enough/possibly omitted.
  - The specification on the paper would not be enough to reconstruct an image.
  - Possible missing part: ResBlock internals.


# 3. Experiments and results

## 3.1. Experimental setup

- Batch size is set as 37.
- Trained using an NVidia A100 80GB GPU, on [Kalantari HDR Dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) [2].

## 3.2. Running the code

1. Download and unzip the Kalantari datasets (both training and test .zips) into `./dataset/` folder.
    - No need to preprocess the dataset. Our Pytorch dataloader code handles it.
2. Create the conda environment and activate it:
    - `conda env create -f conda-environment.yml`
    - `conda activate mcanet-env`
3. Run `python train.py --batch_size=37 --epoch_count=500 --no-wandb`

## 3.3. Results

![image](https://user-images.githubusercontent.com/7654135/177132059-69e81015-3cb6-4bb0-b246-6c01734603c7.png)


![image](https://user-images.githubusercontent.com/7654135/177132124-1365a12b-4869-4b11-833a-017a6918a90d.png)


<p align="center">
  <img width=500 height=auto src="https://user-images.githubusercontent.com/7654135/177131771-f6a32014-4835-4ac2-a054-51062464d93a.png">
</p>

# 4. Conclusion

Compared to the original work:
  - Lack of discussion regarding the reconstruction part and the ad-hoc sigmoid layer proved fatal.
  - The network might be memorizing the samples (the original work might be as well).
  - We obtained grayish images. We believe that this is due to the insufficient network capacity (reconstruction), or the sigmoid layer followed by the tone-mapping operation killing gradients during training.


# 5. References

1. Yan, Q., Wang, B., Li, P., Li, X., Zhang, A., Shi, Q., ... & Zhang, Y. (2020). Ghost removal via channel attention in exposure fusion. Computer Vision and Image Understanding, 201, 103079.

2. Kalantari, N. K., & Ramamoorthi, R. (2017). Deep high dynamic range imaging of dynamic scenes. ACM Trans. Graph., 36(4), 144-1.

# Contact

- Kadir Cenk Alpay
  - kadircenk[-at-]ceng.metu.edu.tr
  - [kadircenk.com](https://kadircenk.com/)
  
- A. Cem Önem
  - onem[-at-]ceng.metu.edu.tr
  - [user.ceng.metu.edu.tr/~onem](https://user.ceng.metu.edu.tr/~onem/)
