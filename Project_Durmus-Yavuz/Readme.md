
# Contrastive Learning based Hybrid Networks for Long-Tailed Image


This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

Learning discriminative image representations is highly important, especially in long-tailed image classification. Long-tailed classification is a widespread research problem, where the key is to overcome the data imbalance issue. Several methods have been proposed to tackle this issue, such as Data Augmentation, Data Re-sampling, Loss Re-weighting, etc. Recently, Contrastive Learning has shown great promise in unsupervised and supervised learning. Peng Wang et al. proposed "Contrastive Learning-based Hybrid Networks for Long Tailed Image Classification" to explore effective supervised contrastive learning strategies and propose a novel hybrid network structure to learn classifiers. It was published in CVPR2021. Our main aim is to produce HybridSC and HybridPSC results for CIFAR10 dataset from Table1 in the paper.

## 1.1. Paper summary

The authors proposed a hybrid network structure for long-tailed image classification in this work. The network is composed of a contrastive loss for feature training and a cross-entropy loss for classifier learning. By doing this, less skewed features and consequently less biased classifiers are expected to be obtained.

In this paper, the authors argue that the original supervised contrastive loss(SCLoss) staging may not be an optimal choice. In the original work, they learn features using SC Loss first and then freeze them to learn classifiers. The authors demonstrated the advantage of their hybrid network on well-known long-tailed datasets.

They also address the memory consumption issue from the original SC Loss and further propose a prototypical supervised contrastive learning strategy, which learns a prototype for each class. They test their methods on three long-tailed image classification datasets, namely CIFAR10, CIFAR100 and iNaturalist, and stated that proposed hybrid networks can outperform the cross-entropy based counterparts.

# 2. The method and my interpretation

## 2.1. The original method

In the original method, the network comprises supervised contrastive learning (SC Loss) based feature learning branch and a cross-entropy (CE) loss-based classifier branch. In the feature learning branch, a non-linear multiple-layer-perceptron (MLP) with one hidden layer is used to map the image representations into a more suitable representation for contrastive learning. Supervised contrastive loss is applied to the normalized representations. Supervised contrastive loss for anchor $z_i$ is computed as the given formula: 

$L_{SCL}(z_i) =\frac{-1}{|z_i+|}\sum_{z_j \in \{z_i+\}}\frac{exp(z_i.z_j / \tau)}{\sum_{z_k, k \neq i} exp(z_i.z_k / \tau) }$

And the overall loss can be computed as:

$L_{SCL} = \sum_{i=1}^{N}L_{SCL}(z_i)$

where N is the mini-batch size.


The classifier learning branch is more straightforward, applying a single layer to the image representations to predict the class-wise logits used to compute the cross-entropy loss. Since the loss functions have different natures, they have different data sampling strategies. The final loss function for the hybrid network is given as follows:

$L_{hybrid} = \alpha L_{SCL}(B_{SC}) + (1-\alpha) L_{CE}(B_{CE})$

Where $\alpha$ is weighting coefficient which adjusted dynamically in each epoch.

To resolve the memory bottleneck issue and mostly retain the feature learning property of SC loss, the authors propose a prototypical supervised contrastive (PSC) loss. The authors aim to attain a similar goal of SC loss by learning a prototype for each class and forcing differently augmented views of each sample to be close to the prototype of their class and far away from the prototypes of the remaining classes. In PSC Loss, each sample is contrasted against the prototypes of all other classes. If a dataset has $C$ classes, the negative size is $C − 1$. The mathematical representation of PSC Loss is given below:

$L_{PSCL}(z_i) = - \log \frac{exp(z_i.p_{y_i} / \tau)}{\sum_{j=1, j \neq y_i}^{C} exp(z_i.p_j / \tau) }$

where $p_{y_i}$ is the prototype representation of class $y_i$,  and $z_i$ is the normalized representation of $x_i$.


## 2.2. Our interpretation 

In SC loss-based hybrid network implementation, hybrid testing implementation and data sampling methods are not given in detail. Our first interpretation is about data sampling. In the original supervised contrastive loss, they use an extra augmented view of the anchor image as a positive instance. We choose to use the exact setup for the representation learning stage.

In the paper, implementation details of the training stage were given in enough detail; however, the crucial part, testing the hybrid networks, was not mentioned as detailed as the training part. To achieve this work's joint learning feature, we have defined our network accordingly. After each epoch, we tested the model's performance using the classifier branch.

Authors stated that they use ResNet-32 as their shared backbone. In feature learning branch, they use single hidden layer MLP to obtain better features for contrastive learning. The dimensions of the MLP is not clearly stated, so we decided to set the output feature dimension to 128.

This project's biggest challenge was implementing the prototypical supervised contrastive learning branch. The prototypical supervised contrastive loss was only given as a generic formula, and any additional information about how the prototypes of each class are calculated was not mentioned. To tackle this problem, we have decided to generate prototypes in each mini-batch by averaging the features with the same label. The primary purpose of the proposed prototypical supervised contrastive loss method is to address the memory bottleneck issue caused by the original supervised contrastive loss method. However, our implementation does not solve this memory issue; we could not implement prototype calculation properly.


# 3. Experiments and results

## 3.1. Experimental setup
In paper, For both long-tailed CIFAR-10 and CIFAR-100 ResNet-32 is used as backbone network to extract image representation. For both branches, random cropping with size 32 × 32, horizontal flip and random grayscale with probability of 0.2 is used as data augmentation strategy. Batch size is set to 512 for both SC and PSC loss based setups. SC loss. They use SGD with a momentum of 0.9
and weight decay of 1×10 −4 as optimizer to train the hybrid networks. The networks are trained for 200 epochs with the learning rate being decayed by a factor of 10 at the 120th epoch and 160th epoch. The initial learning rate is 0.5. For $\alpha$, they use a parabolic decay wrt. the epoch number, where:

$\alpha = 1 - (\frac{T}{T_{max}})^2$ and $T$ = current epoch number and $T_{max} $ = maximum epoch number. 

In SC Loss, temperature($\tau$) is set to 0.1 and in PSC Loss, $\tau$ is set to 0.1 for CIFAR-100 and 1 for CIFAR10.

Since our computational resources were limited, we had to limit the batch size to 64. We set the learning rate to 0.1 and we use ResNet-34 as our backbone. Remaning setup is set according to the original paper.

To generate long-tailed datasets, we have referred to the cited paper "Learning imbalanced datasets with label
distribution-aware margin loss". We have adapted the code from official github repo of this paper to generate long-tailed CIFAR10 and CIFAR100 datasets. According to the paper, we set the imbalance ratio to 10, 50 and 100.

Our overall structure was implemented upon the original Supervised Contrastive Loss implementation [GitHub page](https://github.com/HobbitLong/SupContrast).  

## 3.2. Running the code

Project Directory:
```
├── dataset_loader.py
├── losses.py
├── main.py
├── networks
│   └── resnet_big.py
└── util.py


```
Requirements:

Install PyTorch if required: 

```bash
pip3 install torch torchvision
```


To train and test the SCLoss with CIFAR10 dataset run:

```
python3 main.py --batch_size 512 --learning_rate 0.5 --temp 0.1 --model resnet34 --dataset cifar10
```



## 3.3. Results

Results on CIFAR10 can be seen below:

| Dataset | Method         | Imbalance Ratio | Best Top-1 Accuracy | Original Results |
|---------|----------------|-----------------|---------------------|------------------|
| CIFAR10 | Hybrid SCLoss  | 10              | 78.77               | 91.12            |
| CIFAR10 | Hybrid SCLoss  | 100             | 73.08               | 81.40            |
| CIFAR10 | Hybrid SCLoss  | 50              |                     | 85.36            |
| CIFAR10 | Hybrid PSCLoss | 10              |                     | 90.06            |
| CIFAR10 | Hybrid PSCLoss | 50              |                     | 83.86            |
| CIFAR10 | Hybrid PSCLoss | 100             |                     | 78.82            |

Table1 : Impemented HybridSCLoss and PSCLoss on CIFAR10 dataset with ResNet-34 as backbone architecture.

From Table 1 , we can state that Hybrid SCLoss obtains promising results. The difference between the original results and our results can be may be due to the number of batch sizes. Because of the computational resource issue, remaining results will be published after completion.

# 4. Conclusion
The paper proposes a novel hybrid network structure being composed of a supervised contrastive loss to learn image representations and a cross-entropy loss to learn classifiers, where the learning is progressively transited from feature learning to the classifier learning to embody the idea that better features make better classifiers.
 In this project we have implemented a hybrid network for long-tailed classification and we obtain promising results, specifically in hybrid supervised contrastive loss. However, our prototypical supervised contrastive loss implementation is not effective as the original implementation. Memory issues can not explored in detail and we will try to produce a better/correct implementation for PSC Loss. 



# 5. References

- Project Paper: [Contrastive Learning based Hybrid Networks for Long-Tailed Image
Classification](https://arxiv.org/abs/2103.14267)
- Supervised Contrastive Learning [Paper Link](https://arxiv.org/abs/2004.11362)
- Supervised Contrastive Learning [GitHub Repository](https://github.com/HobbitLong/SupContrast)
- Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss [GitHub Repository](https://github.com/YyzHarry/imbalanced-semi-self)
# Contact

Feyza Yavuz, feyza.yavuz@metu.edu.tr
Tolunay Durmuş, tolunay.durmus@metu.edu.tr
