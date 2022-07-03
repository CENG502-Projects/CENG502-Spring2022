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
\end{equation}
$$

$$
\begin{equation}
\sum_{\mathcal{V}_{l}^{c r f} \backslash\{u_{i}^{l}\}}
\end{equation}
$$

$$
\begin{equation}
\mathbf{P}\left(u_{1}^{l}, u_{2}^{l}, \ldots, u_{N K+T}^{l} \mid \mathbf{F}^{l}, \mathcal{Y}_{s}\right)
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
