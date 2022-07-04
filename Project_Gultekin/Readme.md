# CLIFFNet for Monocular Depth Estimation with Hierarchical Embedding Loss


This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

CLIFFNet for Monocular Depth Estimation with Hierarchical Embedding Loss is the paper published in ECCV 2020. Authors : Lijun Wang, Jianming Zhang, Yifan Wang, Huchuan Lu, and Xiang Ruan. Main pupose of the paper is depth estimation using single image. NYU-Depth V2 dataset were used in the paper. 

## 1.1. Paper summary

Hierarchical Embedding Loss is proposed. The loss function uses embeding depth feature maps instead of directly using depth map.

In order to produce depth embeddings, Hierarchical Embedding Generator (HEG) proposed. HEG contains two different networks: HEG-S and HEG-R. HEG-S takes depth map as input and makes classification to their scenes using NYU dataset labeled scenes in order to uses similar informations of the same scenes. After HEG-S training, the output feature embedding maps of the intermediate convolutional layers is adopted as hierarchical embeddings. HEG-R is the Network that contains encoder and decoder parts. Depth maps are the input of HEG-R network and and after feature extraction in encoder, depths are reconstructed using decoder network. Depth embeddings features extracted by HEG-R Encoder network can adopted to hierarchical embeddings.

After HEG network generates the embeddings for hierarchical embeding loss, CLIFFNET network makes the depth estimation using RGB single image as input. The outputs of CLIFFNET are depth maps and depth embedding generation required.  

Main Contributions mentioned in the paper: 

"– A new form of hierarchical loss computed in depth embedding spaces is proposed for depth estimation.

– Different architectures and training schemes of hierarchical embedding generators are investigated to find desirable hierarchical losses.

– A new CLIFFNet architecture is designed with more effective cross level
feature fusion mechanism."

# 2. The method and my interpretation

## 2.1. The original method

Original method contains some terms : Hierarchical Embeding Loss, Hierarchical Embedding Generator (HEG),HEG-S,HEG-R, and CLIFFNET

Hierarchical Embeding Loss : in general, standart Loss functions uses difference between ground truth depth map and estimated depth map in monocular depth estimation. Instead using directly depth map, depth embedding features are used to compute Loss function in the paper. 
![loss](https://user-images.githubusercontent.com/48828422/177058130-33ca570b-77a5-4fab-bf20-958e5aed6e27.png)

where d ground truth depth map, d^ estimated depth map, G() depth embeddings generator, w hierarchical loss weights.

Hierarchical Embeding Generator (HEG): Depth embedding generator networks to feed loss function. HEG-S and HEG-R networks used.

HEG-S : takes NYU dataset depth maps as input and classifies them to their scenes using NYU dataset scenes in order to use similar structure and properties of the same scene. Extracted depth embedding features by the intermediate convolutional layers can be adopted to Loss function.

![HEG-S](https://user-images.githubusercontent.com/48828422/177058465-3d4ab32f-31f3-45c8-b6ed-cc8a7366b6a2.png)

HEG-R : contains encoder and decoder networks. Encoder network takes NYU dataset depth map as input to extract depth feature embeddings and decoder network reconstruct depth maps using the embedding features. Depth embedding features extracted by intermediate convolutional layers in encoder network can be adopted to Loss function.

![heHEG-R](https://user-images.githubusercontent.com/48828422/177058641-d517416e-1b50-4de6-a2cb-4668083b3aca.png)

CLIFFNET : Cliffnet is the network that trained for depth estimation from single RGB image. It contains feature extraction, feature pyramid, and fetature fusion (Cliff module) part to estimate depth map from single image.


![Cliffnet](https://user-images.githubusercontent.com/48828422/177058790-c89583d0-9f0d-41c7-af1d-506461dc2869.png)


## 2.2. Our interpretation 

Networks (HEG-R,HEG-S and CLIFFNET) implemented separately and couldn'd merged yet.
Hierarchical Embedding Loss not applied yet.

HEG-S and HEG-R networks are implemented nearly the same with in the paper.

CLIFFNET implemented manually instead using ResNet. My interpretation: 

### Feature Extraction Network:
| Layers  | 1 | 2 | 3 | 4 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Type  | Conv(3x3)  | Conv(3x3) | Conv(3x3)  | Conv(3x3)  |
| Output Channel  | 16  | 32  | 32 | 32 |
| Stride  | 2x2  | 2x2  | 2x2 | 2x2 |

Additionally for each layer Conv(3x3) with stride(1x1) applied separately to feed and make connection with feature pyramid network.

### Feature Pyramid Network:
The network is the completely symmetrical to feature ectraction network but it takes concatenated features that are depth features generated from feature extraction network layers and previous feature-pyramid network layers as input and output of the current layer and previous layer of the network feed Cliffnet Module to fuse the features.

### Cliffnet Module : 

The module takes high level and low level embedings generated in feature pyramid network. Firstly High level embeddings are upsampled using transposed convolutional. Upsampled high level embeddings and low level embeddings are multiplied. Then multiplied embeddings and low_level embeddings are concatenated and conv(3x3) with stride(1x1) applied. After that output of the layer concatenated with upsampled high level embedings and again conv(3x3) with stride(1x1) applied. At the end output of the last conv. layer and upsampled high level embeddings added and used as output of the Cliffnet module.

# 3. Experiments and results

## 3.1. Experimental setup

HEG-S, HEG-R and CLIFFNET networks are implemented separately and couldn't merge yet. Hierarchical Loss couldn't implemented yet.

## 3.2. Running the code

First NYU Dataset downloaded to the same directory with the code examples. Download Link: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

Then code examples can be used separately.

    .
    ├── HEG-S.ipynb                # HEG-S network example
    ├── HEG-R.ipynb                # HEG-R network example
    ├── CliffNet.ipynb                # CLIFFNET network example
    └── README.md
    
## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

Furkan Gültekin

email : fege.gul@gmail.com
