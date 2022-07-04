# Dynamic Inference with Neural Interpreters

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

🔑 Modern neural network architectures are capable of learning the data distribution and generalizes well within the training distribution when large amount of data are supplied. The problem with modern architectures is the lack of interpretation capability:

> At the test time model performance is poor when a data which is drawn from related but different distribution is supplied.

This work presents the Neural Interpreter which constitutes a self-attention based network as a system of modules that are called functions. Input tokens are fed to model and routed through the functions via end to end routing mechanism. Proposed architecture provides capability of computation as an attempt to increase model representation along **depth** and **width**.

## 1.1. Paper summary

🔑 This section covers the fundamental ideas & motivation of the paper as well as the proposed architecture.

### Architecture Overview

This section covers the backbone architecture along with 7 proposed mechanisms:
* ⏯ Scripts
* ⏯ Functions
* ⏯ Type Matching and Inference
* ⏯ ModLin Layers and ModMLP
* ⏯ ModAttn
* ⏯ Line of Code (LOC)
* ⏯ Interpreter

### Input and Output
🔑 Input to the Neural Interpreter is a set of vectors that we denote as  {x<sub>i</sub>}<sub>i</sub>  in which  x<sub>i</sub> ∈  R<sup>din</sup>  and the output is another set of vectors  {y<sub>j</sub>}<sub>j</sub>  where  y<sub>j</sub>  ∈  R<sup>dout</sup>  with the same cardinality as the input set. Neural Interpreter expects image tokens as input rather than images as in the case of **ViT**. Input set additionally contains the one or more learned class tokens that are called CLS tokens.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*jEhvJTwTViwvUeI3yPvgow.png" />
</p>

<p align='center'><b>Figure 1:</b> Neural Interpreter Architecture</p>

### Scripts
🔑 At the backbone, Neural Interpreter consists of ns `Scripts`, in **Figure 1** these scripts are denoted as  `Script1` ,  `Script2`  and  `Script3` . Overall, Scripts takes set of vectors of shape `[Batch x N_tokens x Token_dimension]` and maps it into same set cardinality and shape.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*j1k5I8W9vSlyVIc1nNSThw.png" />
</p>
<p align='center'><b>Equation 1:</b>Neural Interpreter stacks scripts ns times to map one input set to another with the same cardinality and shape</p>

**Role:** Scripts function as independent building blocks that can be dropped in any set-to-set architecture, and Neural Interpreters with a single script can perform well in practice.

### Functions
🔑 Functions are the unit of computations in the entire architecture meaning that crucial progress happens in this unit. Formally, a function can be described with its code and signature as follow:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*5ozH6iir23v8q0dHN8Vyyw.png" />
</p>

<p align='center'><b>Equation 2:</b> Function can be described with its code and signature</p>

As denoted in Equation 2 a function (with function index u) is well defined with two-tuple: (s, u). Let's dive into the meaning of these symbols.

~~~
                                             What are `s` and `c` stands for?
~~~
`Signature` of a function is denoted as `s` and have a similar meaning in programming languages. Signature is a normalized vector and each  functions in the `Script` has its unique signature. By means of this distinction among functions, in `TypeMatching` mechanism input tokens are routed differently to each `function`.

🥇 **Important note:** `Signature` vectors are only shared among function of same types within a script.  

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*u10hOcOauL5fYnodRzVT8g.png"//>
</p>
<p align='center'><b>Figure 2:</b> By the help of function signatures, input tokens are distributed independently to the functions</p>

`Code` of a function is denoted as `c` and it instructs how to process input tokens to the functions. Together with `signature`, it takes role in `TypeMatching` mechanism in order to route input tokens to the functions.

🥇 **Important note:** `Code` of a function is shared across same type functions in a script. 

🥇 **Role:** Functions are vector-valued instructions to other components in
the script. They implement the computational units that can be reused in the computational graph.

### Type Matching and Inference
🔑 Type matching mechanism is at the heart of Neural Interpreters and training stability. If not designed well, mode collapse might occur, meaning that all of the tokens goes only one function or no function takes input (zero-out every token in the mask).

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*laTgjgiphPqUVhfx49me-Q.png"/>
</p>

<p align='center'><b>Figure 3:</b> Type Matching Process</p>

Type matching mechanism can be described best as learning proper routing among input tokens and functions. The way routing occurs relies on masking and operates in 3 steps:


1. First, given a set element $x_i$ , it is processed by an MLP module that outputs its type vector $t_i$.
2. Given a `function` f<sub>u</sub> and its `signature` vector
s<sub>u</sub> ∈ T , the compatibility C<sub>ui</sub> between the function fu and a set element x<sub>i</sub> is determined by the **cosine similarity** between s<sub>u</sub> and t<sub>i</sub>.
3. If  compatibility score is larger than a threshold (τ), f<sub>u</sub> is permitted access to x<sub>i</sub>.

We can describe entire process formally using learnable parameter σ and hyperparameter τ as the following:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*UlKTOSTyUuvvjicdh593Ig.png"/>
</p>

<p align='center'><b>Equation 3:</b> TypeInference yields type vector in space T and distance between signature of the function and type vector is calculated via cosine distance.</p>

As it can be seen in **Equation 3** type vector $t_i$ is obtained via MLP layer that is called `TypeInference`.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*slym2bo7Fvh1vFk_2-65eg.png"/>
</p>

<p align='center'><b>Equation 4:</b> Compatiblity score of a token is calculated via negative exponentiation of distance if distance is larger than hyperparameter tau else it is 0. Then Softmax operation is applied to Compatibility scores.</p>

🔓 The compatibility matrix $C_{ui}$ ∈ [0, 1] will serve as a modulation mask for the self-attention mechanism in the `interpreter`.

🥇 **Role:** The `TypeMatching` mechanism is responsible for routing
information through `functions`. The `truncation parameter` τ controls the amount of sparsity in routing.

### ModLin Layers and ModMLP

🔑 ModLin layer is a Linear layer conditioned on code vector. It takes input tokens x and code vector c and performs element-wise fusion operation followed by linear projection as described below:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*jQVXSvB0tUFgHmrhzC4SNQ.png"/>
</p>


<p align='center'><b>Equation 5:</b> In ModLin layer, input tokens are element-wise prodcucted with projected code vectors, again projection occurs in demanded dimensional space.</p>

Further, one may stack the ModLin layers conditioned on the same code vector **c**, which ends up being called **ModMLP**

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*NZids1qWs3MkzcsyJZcOHA.png"/>
</p>

<p align='center'><b>Equation 6:</b> ModMLP Layer uses ModLin layers + GELU activation function as building blocks.</p>

🥇 **Role:** `ModLin layers` and the `ModMLP` can be interpreted as programmable
neural modules, where the program is specified by the condition or code vector **c**.

### ModAttn

🔑 As discussed before, Neural Interpreter is a self-attention based network. ModAttn is a conditional variant of self attention.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*jamAMvWvSp4FAlQ2L9rySg.png"/>
</p>

<p align='center'><b>Figure 4:</b> LOC Layer consists of ModAttn and ModMLP Layer</p>

In this case, conditional vector is the `function code vector`. Under the light of this, we can deduce **Key**, **Query** and **Value** vectors are as follows:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*2IpsLrgKZLyHcCoHai6D2g.png"/>
</p>
<p align='center'><b>Equation 7:</b> Computation of Key, Query and Value vectors are conditioned on function code vector.</p>

Let's make it clear at this point the notation used in **Equation 7**: k<sub>uhi</sub> means key vector for `function: u` `attention head: h` and calculated via `x: i`, x<sub>i</sub>. Same notation applies for others.

Next, given the **keys**, **queries** and the function-variable compatibility matrix $C_{ui}$, the modulated self-attention weights $W_{uhij}$ are given by:

<p align='center'><b>Equation 8:</b> Weight calculation for Attention </p>
<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*hWuTZPskp1FCfrbgWTlzjA.png"/>
</p>

Here, the quantity W<sub>uhij</sub> denotes the attention weights in function f<sub>u</sub> between elements x<sub>i</sub> and x<sub>j</sub>
at head `h` and the softmax operation normalizes along `j`; intuitively, information about x<sub>i</sub> and x<sub>j</sub> is
mixed by f<sub>u</sub> at head h only if W<sub>uhij</sub> different from 0.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*_DkPT0vUMVLBvZ0h95qHGQ.png"/>
</p>
<p align='center'><b>Equation 9:</b> Outputs of attention heads are mixed via ModLin </p>

🥇 **Role:** ModAttn enables interaction between
the elements of its input set in multiple parallel streams, one for each function. The query, key, value,
and output projectors of each stream are conditioned on the corresponding code vectors, and the
interaction between elements in each stream is weighted by their compatibility with the said function.

### LOC Layer
🔑 An **LOC** layer is a **ModAttn** layer followed by a **ModMLP** layer as shown in **Figure 4**, where both layers share the same condition vector $c_u$, and there are weighted residual
connections between the layers. Assuming inputs $\{x_{ui}\}_{u,i}$ to the LOC, we have:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*_PVnHsKEp-gRCB4kIm8OzQ.png"/>
</p>

<p align='center'><b>Equation 10:</b> Residual connections </p>

🥇 **Role:** Role: A **LOC** can be thought of as multiple instances of a layer in the original transformer architecture
(comprising a self-attention and a MLP module with residual connections), applied in parallel streams,
one per function. Computations therein are conditioned on the respective code and signature vectors.

### **Interpreter**

🔑 The interpreter layer is a stack of `nl` LOCs sharing the same function codes $c_u$ and
function-variable compatibilities $C_{ui}$. Assuming the input to the interpreter is a set $\{x_i\}_i$, we have:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*Js3R5ofMmuZjkj5LVeF_2w.png"/>
</p>

<p align='center'><b>Equation 11:</b> Pooling the output of LOC Layers</p>
🥇 **The interpreter** broadcasts a given set element to multiple parallel computational streams,
one for each function. After the streams have processed their copy of the input element, the results are
aggregated by a weighted sum over the streams, where the weights correspond to the compatibility
of the input element with the respective function


# 2. The method and my interpretation

## 2.1. The original method
🔑  The original method relies on `sparsity`: Not all the functions take entire tokens but they are specialized in one function iteration (remember that input tokens are routed independently). This sparsity allows Neural Interpreter functions not only learn the underlying distribution but also interpret how the distribution is generated and when data coming from related distribution is fed to model, model is able to predict from which distribution it is coming from and its class correctly. See Multiclass Classification Experiment for further details.

## 2.2. Our interpretation 
🔑  We strictly stick to the original method: We shared the `code` vector across same type of functions. To avoid the mode collapse (routing input tokens only one function) we defined `signature` vectors from the highest entropy distribution: Normal distribtion and it is fixed. Authors proposed two distinct methods to create signature vectors: 

1. Learnable
2. Fix, initialized from high entropy

To avoid overhead we used fixed implementation. Further, as described in the paper, we shared the **W<sub>c</sub>** across scripts (each script has its own **W<sub>c</sub>** and different scripts have different **W<sub>c</sub>**) and **W** & **b** across all interpreter layers.  `code` vector and `signature` vector is used to determine routing. In our implementation, we further map the range of cosine distance to [0, 1] range in order to avoid `nan` values. (infinity values at exponentials 0 values at mask and their multiplication becomes nan).  

# 3. Experiments and results

## 3.1. Experimental setup
<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*F1wPk37NPc94fHuae9XTSA.png"/>
</p>
<p align='center'><b>Figure 5:</b>  Samples from the Digits Dataset. </p>

🖊 We experimented with **Multi-class Classification Task** in which authors merged 3 different datasets to create **digits** dataset:
~~~
                                                1. MNIST
                                                2. SVHN
                                                3. MNIST-M
~~~
We conducted 2 types of experiments and differenet from the original paper, we used only the following datasets:

~~~
Experiment 1:
                                                1. MNIST
                                                2. SVHN
~~~

In this set-up, MNIST data have the size of [ 1 x 28 x 28 ] and SVHN dataset is of size [ 3 x 32 x 32 ]. To dataloader be compatible and stackable, we resized the MNIST data to [ 32 x 32 ] and repeated its only channel across RGB channels to yield [ 3 x 32 x 32] dimension at the end.

Run the following command to see the training distribution for our implementation:


```Python
from utils import visualize_data

# Parameters for dataset
datasetname = 'digits'
root = 'data/'
batch_size = 128
transform = transforms.Compose([
                  transforms.Resize((32, 32)),
                  transforms.ToTensor()])
loader = get_data_loader(datasetname, root, batch_size, transform)

# Visualize the training distribution
visualize_data(loader)
```

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*4by-qkAQX7mK2ihGZVki6w.png"/>
</p>
<p align='center'><b>Figure 6:</b>  Our Experiment samples from the Digits Dataset. </p>



~~~
Experiment 2:
                                                1. MNIST
~~~
Again running the same scripts for MNIST dataset yields:

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*fEwjutSzo9wgbnQSHPASpw.png"/>
</p>
<p align='center'><b>Figure 6:</b>  Our Experiment samples from the Digits Dataset. </p>

## 3.2. Running the code

Project Directory:
```
├── models
│   └── basic_layers.py
│   └── interpreters.py
│   └── classification_head.py
├── dataset.py
├── utils.py
├── train.py
├── Report.ipynb
├── requirements.txt
├── Readme.md
```

Running the code:
```Python
python3 train.py
```


## 3.3. Results
Original paper obtains comparable result with the state of the art architectures, see table below from original paper:
<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*5BuF92-iRZfSJFzfg41Weg.png"/>
</p>
<p align='center'><b>Figure 7:</b> Multiclass Classification Experiment Results of Original Paper </p>

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/720/0*WK4cDWDCKc6pq9it.png"/>
</p>
<p align='center'><b>Figure 8:</b> Multiclass Classification Experiment Results of Our Implementation </p>

Our implementation yields 10% accuracy on 10 class classification. To investigate the resons in detail, see Challenges section at Conclusion.

# 4. Conclusion


### Experiments & Analysis

In the paper, authors have designed several experiments to speculate the reusable computational units, information modularization, out of distribution functions, and systematic generalization. In this regard, they have designed a toy Boolean experiment, multi-task image classification, and abstract reasoning experiments.

Due to our familiarity with the image classification task we have decided to focus on this part of the experiments. In the original experiment, they have run 3 different classifications i.e. multi-task with 3 different classifications head at the end of neural interpreter model to show the generalizability of neural-interpreter. However, in order to first successfully run and test our model, we have decided to start with a simpler experiment with only 1 classification on MNIST and then increase the difficulty as we succeed in different experiment configurations. As a result for MNIST we have integrated 1 cls token and 1 classification head and  we have kept the model parameters around 0.2-0.3 M as compared to 0.6 M for the multi-task experiment.


### Comparative Discussion & Challenges

In the classification task, we have obtained around 15 % accuracy which highly indicates a training problem when compared to 98-99 % for MNIST classification with many Conv based architectures and also Multi-Task results of neural-interpreter paper. We have tracked our shared parameters, layer matrices, and gradients. Initially our model’s accuracy was below 10 % and we have detected that our compatibility matrices are were going to NAN values, this was mainly due to the exponentiation and masking, where high values for the exponentiation went to infinity and zero values for the mask resulted indeterminacy, NANs which have diverged our models training and blocked gradient flows to our model. This problem directly corresponds to equation-3 and we have fixed this issue by introducing a softmax in between and then L1 norm as our own way of stabilization which allowed our model to reach 15 % of accuracy.

However, this change has direct implications for certain hyperparameters that are suggested for the problems at hand. We have observed that our compatibility matrix was consisting of mostly zeros due to the changed interval for the tau parameter which masks the compatibility scores. As a result we have hand-crafted this hyperparameter to keep sparse but also not vanishing connections between tokens and functions.

Additionally, authors emphasize ‘edge of chaos’ for their model where certain hyperparameters are strictly required for successful training and neural-interpreter is very sensitive to some of them especially i.e. tau which determines the compatibility matrix. Also authors do not explicitly indicate what some of the hyperparameters correspond to i.e. the intermediate feature dimension and this has left us some freedom to make some assumptions about the model and it might be the case that this model configuration has brought us a completely different loss and hyperparameter space which has diminished our model’s sustainability under this configuration.

Unfortunately, we could not tune our hyperparameters in detail with wandb sweep and we could only try some of the hyperparameters manually i.e. tau, nf, and ni. The reason was because the model has significant computational overhead. Although tons of parameters are shared, they are reused tons of times yielding tons of computations. As a result even 1 epoch on MNIST has taken significant time within our limited computational resources.


### Pros & Cons

Neural-interpreter is a generic architecture which grounds on programming language properties: interpreters, codes, types etc. Model learns useful building block functions and stacks them in a certain hierarchy as in a programming language for reuse and abstraction.
This framework promotes architecture adaptation and takes transfer-learning to another level. Models can be extended or simplified by adding or removing functions and they can still generalize to the problem at hand with its set of building blocks as some of these functions maybe a bit redundant or  the extending function may bring new properties.

Parameter sharing allows the model to be kept at low model capacity and the hierarchical way of parameter reuse allows more model complexity as a result. However, too many parameters are shared and although the model capacity is low, the number of computations is significant which we have observed this bottleneck in our experimentations even on a trivial dataset like MNIST.

Additionally, training stability is problematic for this model and model is very sensitive to certain hyperparameters. Also, same learning rate is applied to several shared parameters at different scopes, but they accumulate gradients at different rates which brings a some instability for training. We have faced some difficulties with gradients in our experiments.


# 5. References

[1]	N. Rahaman et al., “Dynamic inference with Neural Interpreters,” arXiv [cs.LG], 2021.

[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint
arXiv:2010.11929, 2020.


Cosine Learning Rate Scheduler adapted from: https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py



# Contact

Alpay Özkan, alpay.ozkan@metu.edu.tr
Hıdır Yeşiltepe, hidir.yesiltepe@metu.edu.tr
