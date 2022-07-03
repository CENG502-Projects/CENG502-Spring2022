# Mutual CRF-GNN for Few-shot Learning

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

We chose Mutual CRF-GNN for Few-shot Learning  to implement as a term project within the scope of the CENG502 Advanced Deep Learning course. This paper is CVPR 2021 paper. We aim to achieve the same results as researchers by implementing this paper, which does not have open source code. Thus, we hope to create an open source for those who want to apply the method in the paper.

## 1.1. Paper summary

This paper brings a new method to GNN. Affinity is a critical part of GNN, and in general affinity is computed in feature space, and 
does not take fully advantage of semantic labels associated to these features.In this paper CRF(Conditional Random Field) is used on labels and features of support data to infer a affinity in the label space. This method is called as Mutual CRF-GNN (MCGN). Their results show that this method outperforms state of art methods on 5-way 1-shot and 5-way5-shot setting on datasets miniImageNet, tieredImageNet, and
CIFAR-FS.

# 2. The method and my interpretation

## 2.1. The original method

Few-shot learning aims to learn a model that can generalize well to new tasks with only a few labelled samples. Each few-shot task has a support set S and a query set Q. The support set S contains N classes with K samples for each class it is called as N-way K-shot. 

Let 

$$
\begin{equation}
\begin{aligned}
\mathbf{F}=\left(\mathrm{f}_{1}, \mathrm{f}_{2}, \ldots, \mathrm{f}_{N \times K+T}\right) \in \mathbb{R}^{(N \times K+T) \times p} 
\end{aligned}
\end{equation}
$$

be the collection of N x K + T feature vectors in one few shot task, where p is the feature dimension. Given an input F0 = F and the
associated graph affinity A0 = A, GNN conducts the following
layer-wise propagation in the hidden layers as 

$$
\begin{equation}
\mathbf{F}^{l+1}=\sigma\left(\mathbf{D}^{-1 / 2} \mathbf{A}^{l} \mathbf{D}^{-1 / 2} \mathbf{F}^{l}\right)
\end{equation}
$$

### Introducing CRF to GNN:
To produce the affinities A that consider contexts, this paper utilizes the marginal distribution of each random variable in CRFs to compute affinity in all GNN layers.
#### Unary Compatibility:

$$
\begin{equation}
\psi\left(u_{i}^{l}=m\right)
\end{equation}
$$

Unary compatibility is to describe the relation between the variable ui of support samples and its corresponding observation. Mathematically, it can be formulated as 

$$
\begin{equation}
\left\{\begin{array}{cl} 1-\eta & \text { if } m=y_{i} \\
\eta /(N-1) & \text { if } m \neq y_{i}
\end{array},\right.
\end{equation}
$$


#### Binary Compatibility:

$$
\begin{equation}
\phi\left(u_{j}^{l}=m, u_{k}^{l}=n\right)
\end{equation}
$$

Binary compatibility is to describe the relations between the connected
random variables, uj and uk. Mathematically, it can be formulated as 

$$
\begin{equation}
\phi\left(u_{j}^{l}=m, u_{k}^{l}=n\right)=\left\{\begin{array}{cl}
t_{2, k}^{l} & \text { if } m=n \\
\left(1-t_{j, k}^{l}\right) /(N-1) & \text { if } m \neq n
\end{array}\right.
\end{equation}
$$

#### Marginal Distribution:
The marginal distribution of variable ui is obtained by marginalizing out all random variables other than ui in CRF. They adopt the loopy belief propagation to calculate marginal distribution of each node in CRF.

$$
\begin{equation}
\mathbf{m}_{l, i \rightarrow j}^{r}=\left[\phi\left(u_{i}^{l}, u_{j}^{l}\right)\left(\left(\mathbf{b}_{l, i}\right)^{r-1} \oslash \mathbf{m}_{l, j \rightarrow i}^{r-1}\right)\right]
\end{equation}
$$

$$
\begin{equation}
\mathbf{P}\left(u_{i}^{l} \mid \mathbf{F}^{l}, \mathcal{Y}_{s}\right) \propto
\sum_{\mathcal{V}_{l}^{c r f} \backslash\{u_{i}^{l}\}}
\mathbf{P}\left(u_{1}^{l}, u_{2}^{l}, \ldots, u_{N K+T}^{l} \mid \mathbf{F}^{l}, \mathcal{Y}_{s}\right)
\end{equation}
$$

where r denotes the round index of belief propagation and r in the range 0 to R. R as the maximum round number, m is the message from ui to uj.

#### Affinity:
Marginal distribution  integrates both the contextual information in CRF and label information of support samples so to estimate affinity matrix,  marginal distribution is used.

$$
\begin{equation}
\hat{a}_{i j}^{l}=\mathbf{P}\left(u_{i}^{l}=u_{j}^{l}\right)=\sum_{m=1}^{N} \mathbf{P}\left(u_{i}^{l}=m\right) \mathbf{P}\left(u_{j}^{l}=m\right)
\end{equation}
$$

### Mutual CRF-GNN:
Mutual CRF-GNN (MCGN) enables GNN and CRF to help each other. For GNN, CRF provides valuable affinity A for feature transformation F. For
CRF, GNN provides better features F for inferring affinity A.

#### Initialization:
Given the images in the support set and the query set, the raw feature F is extracted by a CNN-based feature extractor f_emb.

$$
\begin{equation}
{F}^{1}=f_{e m b}(\mathcal{X}),
\end{equation}
$$

The initial affinity matrix A0 in GNN is initialized by semantic labels from the support set, 

$$
\begin{equation}
a_{i j}^{0}= 
\end{equation}
$$

$$
\begin{equation}
a_{i j}^{0}=\left\{\begin{array}{cc}
1 & \text { if } y_{i}=y_{j} \text { and } i, j \leq N \times K, \\
0 & \text { if } y_{t} \neq y_{j} \text { and } i, j \leq N \times K, \\
0.5 & \text { otherwise, }
\end{array}\right.
\end{equation}
$$

#### Feed-forward Implementation of MCGN:

• Step1: Given the affinity $A_l-1$ and output features $F_l$ from $(l−1)$-th iteration, we estimate the unary and binary compatibility in the CRF. The estimated compatibility functions define the affinities between two connected random variables in CRF.

• Step2: The marginal distribution for random
variables in CRF is inferred by loopy belief propagation, using the compatibility functions obtained from Step 1 and the labels of samples in the support set.

• Step3: The affinities $A_l$ in GNN is derived from the marginal distributions obtained in step 2.

• Step4: The output features $F_{l+1}$ of the $l$-th iteration are computed by aggregating their neighboring features with $A_l$ as their weights.

Repeat above process layer by layer for $L$ iterations and
get the final output $F_{L+1}$ and affinity matrix A_L for network
optimization and inference.

### Loss:

$$
\begin{equation}
\begin{aligned}
\mathcal{L}^{\epsilon r f} &=\sum_{1-N \times K}^{N \times K+T} \sum_{l=1}^{L+1} \mu_{l}^{c r f} \mathbf{C E}\left(\mathbf{P}\left(u_{1}^{l} \mid \mathbf{F}^{l}, y_{0}\right), y_{i}\right), \\
\mathcal{L}^{g n n} &=\sum_{1=N \times K}^{N \times K+T} \sum_{j=1}^{N \times K} \sum_{l=1}^{L} \mu_{l}^{g n n} \mathbf{B C E}\left(a_{i j}^{l}, c_{i j}\right)
\end{aligned}
\end{equation}
$$

where CE indicates the cross entropy, ${\mu_{l}^{c r f}}$ is the weights of each layer; 
$BCE$ indicates the binary cross entropy loss, $\mu_{l}^{g n n}$ is the weights of each layer, 
$c_{i j}$ is 1 if $y_i = y_j$ and 
0 if ${y_i \ne y_j}$.

The total objective function can be a weighted summation of two losses, 

$$
\begin{equation}
\mathcal{L}=\lambda_{\operatorname{cr} f} \mathcal{L}^{\operatorname{cr} f}+\lambda_{g n n} \mathcal{L}^{g n n}
\end{equation}
$$


@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

Sena Eşme senaesme@hotmail.com

Batuhan Vardar batuhanvardar5@gmail.com
