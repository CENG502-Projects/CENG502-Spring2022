# WB-DE⫶TR: Transformer-Based Detector without Backbone

# 1. Introduction

This project aims to reproduce the results presented in the in the paper  [WB-DETR: Transformer-Based Detector without Backbone](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_WB-DETR_Transformer-Based_Detector_Without_Backbone_ICCV_2021_paper.pdf) by Fanfan Liu, Haoran Wei, Wenzhe Zhao, Guozhen Li, Jingquan Peng, Zihao Li. [1]

## 1.1. Paper summary

The first pure-transformer detector WB-DETR (DETR-Based Detector without Backbone) is only composed of an encoder and a decoder without any CNN-based backbones. Instead of utilizing a CNN to extract features, WB-DETR serializes the image directly and encodes the local features of input into each individual token. Besides, to allow WB-DETR better make up the deficiency of transformer in modeling local information, a LIE-T2T (Local Information Enhancement Tokens-to Token) module is designed to modulate the internal (local) information of each token after unfolding. Unlike other traditional detectors, WB-DETR without backbone is more unify and neat. Experimental results demonstrate that WB-DETR, the first pure-transformer detector without CNN, yields on par accuracy and faster inference speed with only half number of parameters compared with DETR[4] baseline.

# 2. The method and my interpretation

## 2.1. The original method

<p align="center">
  <img width="686" alt="Screen Shot 2022-07-02 at 15 21 00" src="https://user-images.githubusercontent.com/43934455/177000539-e642b7ed-bd16-4fde-aad6-b45870a73eef.png" style="width:600px;"/>
</p>


After the exploration of the success of the transformers it is applied to the object detection area. In this area, unlike previous CNN-based works, DETR[4] is introduced as a transformer-based detector with CNN backbone. 
We also know vision transformers[3] which conduct sequence modeling of the patches and still have worse performance compared with CNNs. Because the simple tokenization of the input images fails to model the important local structures like edges or lines. 
Tokens-to-token Vision Transformer (T2T-Vit)[2] solves the problem of  vision transformers by recursively aggregating neighboring tokens into one token. However in T2T the local information in each token and the information between adjacent tokens were not modeled well. 
So in this paper they propose Local Information Enhancement T2T module which not only reorganizes the adjacent tokens but also uses attention on channel-dimension of each token to enhance the local information.


### 2.1.1 Image To Tokens

<p align="center">
  <img width="545" alt="Screen Shot 2022-07-02 at 15 14 03" src="https://user-images.githubusercontent.com/43934455/177000325-4f4a8bad-af3f-4ab7-a987-dc3e76d41251.png" style="width:600px;"/>
</p>

The process of Image to Tokens. Take an input image with 512×512×3 as an example. Firstly, the image is cut to 1024 patches with the size of 32×32 × 3. Then, each patch is reshaped to one-dimensional. Finally, a trainable linear projection is performed to yield required tokens.

They follow the ViT to handle 2D images. Firstly, They cut the image to a size of ($p, p$) with a step size of ($s, s$). In this way, the input image $x \in R^{h \times w \times c}$ is reshaped into a sequence of flattened 2D patches $x_p \in R^{l \times c_p} , where $h$ and $w$ are the height and width of the original image, $c$ is the number of channels, and $l$ represents the length of patch. Among them, $l= \frac{h \times w}{s^2}$, $c_p = p^2 \times c$.
$l$ also serves as the effective input sequence length for the transformer encoder. Their LIE-T2T encoder employs constant latent vector size $d$ through all of its layers. And thus, they flatten and map the patches to d dimensions with a trainable linear projection. More specifically, this linear projection has an input and output dimensions of $c_p$ and $d$, respectively. They name the output of this projection as the tokens $T_0$.

### 2.1.2 LIE-T2T encoder
<p align="center">
  <img width="854" alt="Screen Shot 2022-07-02 at 15 42 08" src="https://user-images.githubusercontent.com/43934455/177001196-3da728a0-fd3e-46f0-aac5-284067a6e864.png" style="width:600px;"/>
</p>

After the process of image to tokens, they add positional encodings to target tokens to make them carry location information. Then, the resulting sequence of embedding vectors serves as input to the encoder, as shown above. Each encoder layer keeps a standard architecture which consists of a multi-head self-attention module and a feed forward network (FFN). An LIE-T2T module is equipped behind each encoder layer to constitute the LIE- T2T encoder. The LIE-T2T module can progressively reduce the length of tokens and transform the spatial structure of the image.
Since they do not use any CNN-based backbone to extract image features, instead of directly serializing the image, the local information of the image is encoded in each independent token. 

<img width="466" alt="Screen Shot 2022-07-02 at 15 42 44" src="https://user-images.githubusercontent.com/43934455/177001213-c24de276-4e64-4ee7-ab49-7926ca47b0b1.png" align="left" style="width:300px;"/>

Concretely, LIE-T2T module calculates attention on the channel-dimension of each token. The attention is calculated separately for each token. More detailed iterative pro- cess of LIE-T2T module is shown in Figure 5, which can also be formulated as follows:
- $T$ = $Unfold$($Reshape$($T_{i}$))                              
- $S$ = $Sigmoid$ ($W_{2}$ · ReLU ($W_{1}$ · $T$ ))         
- $T_{i}$+1 = $W_{3}$ · ($T$ · $S$)     

where Reshape means the operation: reorganize ($l_{1}$ × $c_{1}$) tokens into ($h × w × c$) feature map. Unfold represents stretching ($h$ × $w$ × $c$) feature map to ($l_{2}$ × $c_{2}$) tokens. $W_{1}$ , $W_{2}$, and $W_{3}$ indicate parameters of corresponding fully connected layer. They use the ReLU activation to find its nonlinear mapping and employ the Sigmoid function to generate the final attention. The input of the LIE-T2T encoder is with the dimension of (($h/s$ × $w/s$) × 256).

<br clear="left"/>

#### 2.1.2.1 T2T vs LIE-T2T 

<p align="center">
  <img width="500" alt="Screen Screen Shot 2022-07-03 at 19 17 46" src="https://user-images.githubusercontent.com/43934455/177048252-3b67809a-da21-4f84-b146-964cb4ed45ad.png" style="width:600px;"/>
</p>

T2T[2] aggregates the information of adjacent tokens through reshape and unfold operations. Based on T2T, LIE-T2T[1] can realize local spatial attention of reshaped $T_{i}$ by calculating channel attention of unfolded $T_{i+1}$ . $F (·, W )$ means an attention calculation, $F (·)$ represents element-wise multiplication and F C indicates the FC layer.
As we all know, the self-attention of transformer has strong global information modeling ability, which can commendably modulate the contexts between different tokens. However, the local information in each token and the information between adjacent tokens in space are not well modeled. In other words, transformer lacks the ability of local information modeling. Although the T2T [2] module can aggregate the contexts of adjacent tokens, it is unable to model the internal information of the aggregated independent token separately, as illustrated in above figure (a). Accordingly, LIE-T2T (Local Information Enhancement-T2T) as shown in above figure (b), not only reorganizes and unfolds the adjacent tokens, but also calculates the attention on the channel-dimension of each token after unfolding. Because the tokens are obtained from feature map through unfold operation, modeling the relationship between channels of the tokens is equivalent to modeling the spatial relationship between the pixels in feature map. That is why channel attention in LIE-T2T can enhance local information.

Additionally, as the length of tokens in the T2T module[2] is larger than the normal case (16 × 16) in ViT[3], the MACs and memory usage are huge. To address the limitations, in our T2T module[2], we set the channel dimension of the T2T layer small (32 or 64) to reduce MACs, and optionally adopt an efficient Transformer such as Performer[5] layer to reduce memory usage at limited GPU memory. 

Performers[5] are a new class of models and they approximate the Transformers. They do so without running into the classic transformer bottleneck which is that, the Attention matrix in the transformer has space and compute requirements that are quadratic in the size of the input, and that limits how much input (text or images) you can feed into the model. Performers get around this problem by a technique called *Fast Attention Via Positive Orthogonal Random Features (FAVOR+)*. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence, and low estimation variance. Performers; capable of provably accurate and practical estimation of regular (softmax) full-rank attention, but of only linear space and time complexity and not relying on any priors such as sparsity or low-rankness. Performers are the first linear architectures fully compatible (via small amounts of fine-tuning) with regular Transformers.



## 2.2. Our interpretation 

In our interpretation, mainly original [DETR](https://github.com/facebookresearch/detr) code is used for transformer decoder model initialization and forward, model parallelizing, training, evaluation, data read, data augmentation etc. All of our changes are applied to the backbone part of the code. For token transformer and token performer parts are taken from [T2T-ViT code](https://github.com/yitu-opensource/T2T-ViT) and our image to token and LIE-T2T implementations are inspired from the same code. In Section 3.2 of the paper, positional encoding process is introduced as a 1D process, in contrast to the 2D positional embedding in DETR. For that, Positional encoding code of DETR[4] paper is adapted to 1D by us. 

In the paper, none of the kernel sizes, padding or stride of any Unfold layer is explained. Kernel sizes are assumed by 3 and padding is assumed as 1 since they are the most common options in a backbone. However, larger kernel sizes may affect the model like a deeper network and it may create a performance difference with better/worse results. Stride is selected as 2 for first M layers and 1 for the rest, and M is selected as 5 in order to match 32 step size. Lower step size causes exponential GPU usage for our case, which causes CUDA out of memory error. Higher step size causes exponentially worse detection performence. 

# 3. Experiments and results

## 3.1. Experimental setup

The paper introduces their experimental setup as follows: The main settings and training strategy of WB-DETR are mainly followed by DETR[4] for better comparisons. All transformer weights are initialized with Xavier Init, and our model has no pre-train process on any external dataset. By default, models are trained for 500 epochs with a learning rate drop 10× at the 400 epoch. We optimize WB-DETR via an Adam optimizer with a base learning rate of 1e−4 and a weight decay of 0.001. We use a batch size of 32 and train the network on 16 V100 GPUs with 4 images per-GPU. We use some standard data augmentations, such as random resizing, color jittering, random flipping and so on to overcome the overfitting. The transformer is trained with a default dropout of 0.1. We fix the number of decoding layers at 6 and report performance with the different layer number N and K of encoder: When N and K is n and k, the corresponding model is named as $WB-DETR_{nk}$.

We have tried to follow the same experimental setup as explained in the paper, except the batch size. Since we have used a TUBITAK TRUBA instance for intensive training, we have used 8 NVIDIA A100 GPUs with 12 images per-GPU. The batch size explained in the paper's setup section is not clear, as they  firstly say that their batch size is 32, then 4 images for each of 16 GPUs (64). In any way, we were able to use higher batch size (96), in order to have faster convergence in a limited time.

## 3.2. Running the code

There are no extra compiled components in WBDETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:

install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

### 3.2.1 Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

### 3.2.2 Training
To train baseline WBDETR on a single node with 8 gpus for 500 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --cfg config.yaml
```

### 3.2.3 Evaluation
To evaluate WB-DETR(2,8) on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --eval --resume checkpoint.pth --cfg config.yaml
```

## 3.3. Results

PyTorch training code and pretrained models for **WB-DETR**.

Our code is able to any (N,K) pair in WB-DETR(N-K) experiments. Yet, due to the limitation in computational resources only two experiments are done. One of it is WB-DETR (0-4) for ~350 epochs and the other is WB-DETR(2-8) for ~500 epochs. 

We provide baseline WB-DETR models.
AP is computed on COCO 2017 val5k.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>patch</th>
      <th>step size</th>
      <th>epochs</th>
      <th>AP 0.5:0.95 (%)</th>
      <th>AP 0.5 (%)</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>WB-DETR(2-8) paper</td>
      <td>32</td>
      <td>32</td>
      <td>500</td>
      <td>33.9</td>
      <td>61.0</td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>WB-DETR(2-8) ours</td>
      <td>32</td>
      <td>32</td>
      <td>500</td>
      <td>22.3</td>
      <td>38.0</td>
      <td><a href="https://users.metu.edu.tr/aybora/data/checkpoint.pth">model</a>&nbsp;|&nbsp;<a href="https://users.metu.edu.tr/aybora/data/log6.txt">logs</a></td>
    </tr>
    <tr>
      <td>2</td>
      <td>WB-DETR(0-4) ours</td>
      <td>32</td>
      <td>32</td>
      <td>341</td>
      <td>14.4</td>
      <td>27.0</td>
      <td><a href="https://users.metu.edu.tr/aybora/data/checkpoint2.pth">model</a>&nbsp;|&nbsp;<a href="https://users.metu.edu.tr/aybora/data/log2.txt">logs</a></td>
    </tr>
  </tbody>
</table>

### 3.3.1 T2T vs LIE-T2T

Here is the comparison of WB-DETR(2-8) (LIE-T2T) and WB-DETR(0-4) (T2T) models.

**AP metric**

<img width="706" alt="Screen Shot 2022-07-03 at 19 01 49" src="https://user-images.githubusercontent.com/43934455/177047680-b63060f2-695a-41ae-96fb-f880c5069182.png">

**AP_50 metric**

<img width="706" alt="Screen Shot 2022-07-03 at 19 04 03" src="https://user-images.githubusercontent.com/43934455/177047742-d6efebdd-0629-40c9-9629-6732949e8496.png">

**Classification Error**

<img width="706" alt="Screen Shot 2022-07-03 at 19 05 28" src="https://user-images.githubusercontent.com/43934455/177047786-2347e059-a528-4a9e-9b61-dc6a63e7bb13.png">

**Loss**

<img width="706" alt="Screen Shot 2022-07-03 at 19 06 41" src="https://user-images.githubusercontent.com/43934455/177047846-4509b4f1-c7ab-4e56-bd99-23081da8d1b3.png">

According to the results, even though it is not fair to compare with different number of layers and epochs, it can be seen that LIE-T2T outperforms the original T2T by enchancing local information. For LIE-T2T, learning rate is dropped at 400th epoch and it increases the performance even more. 

# 4. Conclusion

Our results are worse than the proposed results. There might be a couple of differences between the original method and our interpretation such as kernel sizes, padding or stride of Unfold layers. It is not clear that if the original method uses transformer or performers, we used performer in LIE-T2T encoder because of limitation in hardware. We did our experiments using 32 step size because lower step size causes exponential GPU usage for our case, which causes CUDA out of memory error. Higher step size causes exponentially worse detection performence. 

# 5. References

- [1] F. Liu, H. Wei, W. Zhao, G. Li, J. Peng and Z. Li, "WB-DETR: Transformer-Based Detector without Backbone," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 2959-2967, doi: 10.1109/ICCV48922.2021.00297.
- [2] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan, "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet", Computer Vision and Pattern Recognition (CVPR) 2021, doi: 10.48550/ARXIV.2101.11986
- [3] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Computer Vision and Pattern Recognition (CVPR) 2021, doi: 10.48550/ARXIV.2010.11929
- [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, "End-to-End Object Detection with Transformers", Computer Vision and Pattern Recognition (CVPR) 2020, doi: 10.48550/arXiv.2005.12872
- [5] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller, "Rethinking Attention with Performers", Machine Learning 2021, doi: 10.48550/ARXIV.2009.14794
# Contact

[Güneş Çepiç](https://github.com/gunescepic), [Aybora Köksal](https://github.com/aybora)

