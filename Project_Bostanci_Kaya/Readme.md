# Towards Better Laplacian Representation in Reinforcement Learning with Generalized Graph Drawing

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing the paper, "[Towards Better Laplacian Representation in Reinforcement Learning with Generalized Graph Drawing](https://arxiv.org/pdf/2107.05545.pdf)" which was published by Kaixin et al. in [ICML2021](https://icml.cc/Conferences/2021/Schedule?). 

See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

In Reinforcement Learning (RL) tasks, there exists an agent-environment interaction mechanism through taking actions and obtaining rewards. Agents interact with an environment and this environment provides numeric reward signals. Main goal of the RL is discovering best actions in order to maximize the cumulative reward.

![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/agentenvironmentaction.png)

Cartesian (x, y) positions of the agent can be used for the state representations. In order to capture the geometry of the underlying state sparce, it is desired to learn a state representation from the environment.

![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/environment.jpg)

A graph $\mathcal{G}=(\mathcal{S},\mathcal{E})$ can be constructed from the state transition process by denoting $\mathcal{S}$ states and $\mathcal{E}$ edges. Then, an adjacency matrix $A$ is defined with state transition probabilities. 

State embeddings are crucial to find the state-space geometry. They can be obtained through the calculations of the graph Laplacian matrix $L$. 

However, directly applying eigen-decompositon to the graph Laplacian matrix is computationally inefficient especially when the number of states is increased. Therefore, an estimation method is required to find the closest representation vectors to the actual eigenvectors. 


## 1.1. Paper summary

### Reinforcement Learning (RL) overview: 

An agent interacts with an environment by observing states and taking actions, with an aim of maximizing cumulative reward

The environment then yields a reward signal Rt sampled from the reward function r(st; at).
The state observation in the next timestep st+1 2 S is sampled according to an environment-specific transition distribution function p(st+1jst; at). A policy is defined as a mapping π : S ! A that returns an action a given a state s.

The goal of the agent is to learn an optimal policy π∗ that maximizes the expected cumulative reward:

### Laplacian representations in RL:

 - Graph definition: $\mathcal{G}=(\mathcal{S},\mathcal{E})$  
 - Vertices $\mathcal{S}$ = States,  
 - Edges  $\mathcal{E}$ = State transitions, 
 - State transition probabilities $P_{ij}$ 
 - Adjacency matrix elements ($A_{ij}$) = $P_{ij}$ 
 - Degree matrix, $D = \text{diag}(A\text{\textbf{1}})$ 
 - Laplacian matrix, $L = D-A$

### Estimation of the Laplacian representations:

We denote the i-th smallest eigenvalue of $L$ as $\lambda_i$, and the corresponding unit eigenvector as $e_i \in \mathbb{R}^{|\mathcal{S}|}$ .

The d-dimensional Laplacian representation of a state s:

$$
\begin{equation}
\begin{gathered}
\varphi(s) = (e_1[s], \cdots , e_d[s])
\end{gathered}
\end{equation}
$$
 
where $ei[s]$ denotes the entry in vector $ei$ that corresponds to state s. 

In particular, e1 is a normalized all-ones vector and has the same value for all s. 

Spectral graph drawing objective which is introduced in [[2]](#2):

$$
\begin{equation}
\begin{gathered}
\min_{u_{1}, \cdots, u_{d}}  \sum_{i=1}^{d} u_{i}^{T} L u_{i} \\
\text{ s.t. } u_{i}^{T} u_{j}=\delta_{i j}, \forall i, j=1, \cdots, d,
\end{gathered}
\end{equation}
$$ 

The authors in [[1]](#1) proposes that the generalized graph drawing (GGD) objective can be approximated as follows:

$$
\begin{equation}
\begin{gathered}
\min_{u_{1}, \cdots, u_{d}}  \sum_{i=1}^{d} c_{i} u_{i}^{T} L u_{i} \\
\text{ s.t. }  u_{i}^{T} u_{j}=\delta_{i j}, \forall i, j=1, \cdots, d
\end{gathered}
\end{equation}
$$ 

Newly introduced coefficients $(c_{i}, \cdots, c_{i})$ are able to produce unique global minimizer.

The authors suggest that the coefficients can be selected as decreasing order such that: $c_{1} = d, c_{1} = d-1, \cdots, c_{i} = d-i, \cdots, c_{d} = 1$ where $i=1, 2, \cdots, d$

Thus, new GGD objective can be written as follows [[1]](#1):

$$
\begin{equation}
\begin{gathered}
\min_{u_{1}, \cdots, u_{d}}  \sum_{i=1}^{d}(d-i+1) u_{i}^{T} L u_{i} \\
\text{ s.t. }  u_{i}^{T} u_{j}=\delta_{i j}, \forall i, j=1, \cdots, d .
\end{gathered}
\end{equation}
$$ 

In this report, the equation above is referred as "Wang's loss".

# 2. The method and my interpretation

## 2.1. The original method

We adapted a baseline code to implement a representation loss which is defined by Wu in [[2]](#2):

```Python
def l2_dist(x1, x2):
    return (x1 - x2).pow(2).sum(-1)

def neg_loss(x, c=1.0, reg=0.0):
    """
    x: n * d.
    sample based approximation for
    (E[x x^T] - c * I / d)^2
        = E[(x^T y)^2] - 2c E[x^T x] / d + c^2 / d
    #
    An optional regularization of
    reg * E[(x^T x - c)^2] / n
        = reg * E[(x^T x)^2 - 2c x^T x + c^2] / n
    for reg in [0, 1]
    """
    n = x.shape[0]
    d = x.shape[1]
    inprods = x @ x.T
    norms = inprods[torch.arange(n), torch.arange(n)]
    part1 = inprods.pow(2).sum() - norms.pow(2).sum()
    part1 = part1 / ((n - 1) * n)
    part2 = - 2 * c * norms.mean() / d
    part3 = c * c / d
    # regularization
    if reg > 0.0:
        reg_part1 = norms.pow(2).mean()
        reg_part2 = - 2 * c * norms.mean()
        reg_part3 = c * c
        reg_part = (reg_part1 + reg_part2 + reg_part3) / n
    else:
        reg_part = 0.0
    return part1 + part2 + part3 + reg * reg_part
```
Common part to calculate the overall representation loss:

```Python
def _build_loss(self, batch):
     s1 = batch.s1
     s2 = batch.s2
     s_neg = batch.s_neg
     s1_repr = self._repr_fn(s1)
     s2_repr = self._repr_fn(s2)
     s_neg_repr = self._repr_fn(s_neg)
     loss_positive = pos_loss(s1_repr, s2_repr)
     loss_negative = neg_loss(s_neg_repr, c=self._c_neg, reg=self._reg_neg)
     loss = loss_positive + self._w_neg * loss_negative
     info = self._train_info
     info['loss_pos'] = loss_positive.item()
     info['loss_neg'] = loss_negative.item()
     info['loss_total'] = loss.item()
     return loss
```
## 2.2. Our interpretation 

We used the following approach to implement the representation loss that is introduced by Wang in [[1]](#1)

```Python
def pos_loss(x1, x2):
    d = x1.shape[1]
    pos_loss = 0
    for dim in range(d, 0, -1):
        pos_loss += (x1[:, :dim] - x2[:, :dim]).pow(2).sum(dim=-1).mean()
    return pos_loss

def neg_loss(x, c=1.0, reg=0.0):
    n = x.shape[0]
    d = x.shape[1]
    neg_loss = 0
    for dim in range(d, 0, -1):
        # # Loss for negative pairs
        inprods = x[:, :dim] @ x[:, :dim].T
        norms = th.diagonal(inprods, 0)
        part1 = (inprods.pow(2).sum() - norms.pow(2).sum()) / (n * (n - 1))
        part2 = -2 * c * norms.mean() / d
        part3 = c * c / d
        neg_loss += part1 + part2 + part3
    return neg_loss
```

# 3. Experiments and results

## 3.1. Experimental setup

All experiments are conducted in two discrete grid world reinforcement learning environments. These environments are generated with MiniGrid which is provided in [[3]](#3).

Two discrete environments, *GridRoom* and *GridMaze* are used in the representation learning and expected reward maximization experiments.

*GridRoom* environment contains 20x20 grid with 271 states and *GridMaze* environment contains 18x18 grid with 161 states. 4 possible actions are available for these environments. An agent is able to go up, down, left and right.

We used position observations as input.

Episode lengths for each environment is 50.

Training parameters:
 - Episode length = 50
 - Batch size = 1024
 - reducedDimension, d =10
 - Adam optimizer with learning rate $10^{-3}$
 - Discount factor = 0.9
 - Number of training steps = 200000
 - MLP networks with RELU activations for representation learning and DQN

In the RL experiments, a mixture of the rewards is used  by following experiments of Wu in [[2]](#2):
**Sparse** = -1[norm(stateMapping-goalVector)$> \epsilon$] 
**Distance** = -norm(stateMapping-goalVector)
**Mixture of the rewards** = $0.5*\text{Sparse} + 0.5*\text{Distance}$

## 3.2. Running the code

We conducted the experiments on the Google's Colab (https://colab.research.google.com/) platform.

### Required packages: 

 - gym3, 
 - minigrid,  
 - YAML (not reqired for Colab)

You can follow the steps in order to understand how to install required packages and how to conduct any experiment.

### Source code directory structure

```
\environments
│
└───\environments\gridRoom
│   │
│   └───\environments\gridRoom\lossWu
│   │   │
│   │   └───\environments\gridRoom\lossWu\run1
│   │   │
│   │   └───\environments\gridRoom\lossWu\run2
│   │   │
│   │   └───\environments\gridRoom\lossWu\run3
│   │
│   └───\environments\gridRoom\lossWang
│       │
│       └───\environments\gridRoom\lossWang\run1
│       │
│       └───\environments\gridRoom\lossWang\run2
│       │
│       └───\environments\gridRoom\lossWang\run3
│   
└───\environments\gridMaze
    │
    └───\environments\gridMaze\lossWu
    │   │
    │   └───\environments\gridMaze\lossWu\run1
    │   │
    │   └───\environments\gridMaze\lossWu\run2
    │   │
    │   └───\environments\gridMaze\lossWu\run3
    │
    └───\environments\gridMaze\lossWang
        │
        └───\environments\gridMaze\lossWang\run1
        │
        └───\environments\gridMaze\lossWang\run2
        │
        └───\environments\gridMaze\lossWang\run3

```

### How to conduct an experiment:

**Step-1**:  Upload \sourceCode to your Google Drive

**Step-2**: Copy the Jupiter notebook to your Google Drive

**Step-3**: Open the notebook in Google Colab

**Step-4**: Follow the instructions on the notebook. These instructions can be summarized as follows:

**Step-4.1**: Install required packages:

```bash
 pip install gym3
```  

```bash
 pip install gym-minigrid
```  
Note that you need to install YAML package after download the package through http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyyaml if you run the code outside of the Colab platform:

```bash
 pip3 install PyYAML-3.11-cp35-none-win32.whl
``` 

**Step-4.2**: Change directory to the spesific experiment. For example, following command can be used for Grid Room experiments with Wang's loss for run1, 
```bash
 cd \environments\gridRoom\lossWang\run1
```   
**Step-4.3**: Train network for state representations: 

```bash
 !python train_laprepr.py --env_id=HardMaze --log_sub_dir=test --args="w_neg=1.0"
```

  **Step-4.4**: Train DQN: 
  
```bash
 !python train_dqn_repr.py --log_sub_dir=mix --env_id=HardMaze --repr_ckpt_sub_path=laprepr/HardMaze/test/model.ckpt --reward_mode=mix
```
  
   **Step-4.5**:  Visualizate representations and plot rewards for each training step: 

```bash
 !python visualize_reprs.py --log_sub_dir=laprepr/HardMaze/test
```  
 

### How to obtain results:

After you run a code with run1 and Wang's representation loss, you can find the results such that:

1. Estimated state representation figures for each dimension: 
```
 \environments\gridMaze\lossWang\run1\log\visualize_reprs\HardMaze_appr_dimension1.pdf
```  
2. Ground-truth state representation figures:
```
 \environments\gridMaze\lossWang\run1\log\visualize_reprs\HardMaze_gt_dimension1.pdf
```  
3. Reward maximization figures: 
```
 \environments\gridMaze\lossWang\run1\log\visualize_reprs\EpisodicReturnVsTrainingSteps.pdf
```  
You can obtain these results for different runs, losses and environments.
You can also observe the pre-trained results. 

## 3.3. Results

### Representation learning results:

![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/ourGridRoomResults.png)
![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/ourGridMazeResults.png)
![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/similarityResults.png)

### Reinforcement learning results: 
![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/ourGridRoomRewards.png)
![alt text](https://github.com/CENG502-Projects/CENG502-Spring2022/tree/main/Project_Bostanci_Kaya/figures/ourGridMazeRewards.png)

# 4. Conclusion

 - All experiments in the paper could not be covered.
 - Obtained rewards per episode during the training iterations not
   given in the paper but in our results it fluctuates
 - Unlike the continuous environments, in discrete environment eigen-
   decompositions are not very problematic because the number of states are
   not too many.
 - Reduced dimension for the Laplacian eigenvectors is chosen as 10 but any detail was not given for this trade
   off in the paper.

# 5. References

<a id="1">[1]</a> 
Wang K. et. al (2021), Towards Better Laplacian Representation in Reinforcement Learning with
Generalized Graph Drawing ([pdf](https://arxiv.org/pdf/2107.05545.pdf)), ICML2021.

<a id="2">[2]</a> 
Wu Y. et. al (2019), The Laplacian in RL: Learning Representations with Efficient Approximations ([pdf](https://arxiv.org/pdf/1810.04586.pdf)), ICLR2019.

<a id="3">[3]</a> 
Chevalier-Boisvert et. al (2018),  Minimalistic Gridworld Environment for OpenAI Gym ([github](https://github.com/Farama-Foundation/gym-minigrid)).

<a id="4">[4]</a> 
Wu Y. (2021), Learning Laplacian Representations in Reinforcement Learning ([github](https://github.com/yifan12wu/rl-laplacian)).

# 6. Acknowledgements

Thanks to Kaixin Wang for his precious help!

# Contact

Safa Mesut Bostancı: [Github](https://github.com/smbostanci)

Semih Kaya: [Github](https://github.com/kayyasemih)