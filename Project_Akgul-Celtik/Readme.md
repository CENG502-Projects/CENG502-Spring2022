# Paper title [Learning Dynamics via Graph Neural Networks for Human Pose Estimation and Tracking]

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction
<!--- @TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility). ---> 
The paper is published for CVPR 2021. The goal of the paper is to find the human poses in given frames by using previous n history frames. Our goal was to train the GNN model of the paper and use pretrained Single Frame Pose Estimator and pretrained Object Detector.

## 1.1. Paper summary

<!--- @TODO: Summarize the paper, the method & its contributions in relation with the existing literature. --->
Multi-person pose estimation and tracking is a hot research area that has been used in many applications to understand the human actions. Most of the appproaches use CNNs in order to detect humans and then generate poses. However, it is possible for them to miss some of the occluded people. This paper proposes to use GNNs to overcome this issue by storing previous frames to predict current (occluded) people's poses. Experiments done on PoseTrack 2017 and PoseTrack 2018 datasets show that the proposed method achieves better performances than the state-of-the-art results on both human pose estimation and tracking tasks.


# 2. The method and my interpretation

## 2.1. The original method

<!--- @TODO: Explain the original method. --->
Paper uses three models to find the human poses:
Single Frame Pose Estimator
GAN Model 
Object Detector
First, humans are found using object detector, where faster-rcnn with feature pyramid network is used.
Second, the found human is cropped and rescaled. Then, Single Frame Pose Estimator is used to find current poses.
Third, GAN model is used on the n previous frames, and it is tried to find the pose in the current frame.
And lastly, Single Frame Pose Estimator output and GAN output are merged to find all of the poses.

![image](https://user-images.githubusercontent.com/64609605/177032733-06c31927-73f5-4117-8e70-e7542c5c6e44.png)
Figure 1. Overall pipeline of the proposed method

The proposed method's overall pipeline is shown in the figure above. The model uses two separate processes to find human poses in current frame _t_ of a video. In the first process, it tries to find the poses as in the classical approaches by using a Single-Frame Pose Estimation model with possible missed human detections. In the second process, it tries to make us of the previous _n_ (default value is 3 in the paper) frames to understand the motion of the people to predict the poses that would be in the next frame using a GNN Pose Prediction model. Then the result of the Single-Frame Pose Estimation and GNN Pose Prediction model are merged to obtain final poses.

A- Single-Frame Pose Estimation

Pose estimation is performed for each frame. Each human detection in a frame is cropped and then rescaled to 384×288. These detections are fed into the HRNet (pretrained on COCO) and it produces 96x72 heatmaps for each 17 joints. However, we ignore the ear joints due to the fact that paper states it uses 15 joints in 2017 dataset format. The positions of the $\textit{k}$-th joints are computed as:


$$\ l^{*}_{k} = \underset{(i,j)}{\arg\max} \textbf{H}_{ijk}$$
   
where $l^{*}_{k}$ is the position within heatmap.
   
The training loss of the single-frame pose estimation model uses these heatmaps. The ground truth heatmaps are generated using the following 2D Gaussian distribution:

$$\textbf{H}_{ijk}^{gt} = exp(-\frac{||(i,j)-l_{k}||^{2}_{2}}{\sigma^{2}})$$

   
$\sigma$ is set to 3 and the model is trained with the following loss:

$$\mathcal{L}_{e} = \sum_i^H\sum_j^W\sum_k^K ||\textbf{H}_{ijk}^{pred} - \textbf{H}_{ijk}^{gt} ||^{2}_{2}$$

where $H$, $W$, $K$ are the height, the width of heatmaps, and the number of joints, respectively.

B- Dynamics Modeling via GNN

The people in history frames and current frame are used on the training of GNN model. It takes the joints in all frames as nodes and creates corresponding edges within a frame and between consecutive frames. This structure aims to understand the motion dynamics, and GNN updates the node features with respect to these dynamics. For pose prediction, use the results of the current frame.
    
As mentioned, joints of the history frames and current frame are used as the nodes of the GNN. Three kinds of cues are used for each joint:
    
* Visiual feature ($v_k$): The visiual features from the backbone CNN of the single-frame pose estimator    
* Joint encoding ($c_k$): The encoding of its joint type with a learnable lookup
table    
* Position feature ($p_k$): 2D position and confidence score from pose estimator
    
All the 2D positions used fır the $p_k$ are normalized with respect to the center of last tracked pose. All of these cues are fed into different MLPs that do not share weights to have the same dimension and are merged using average pooling. Therefore, the final feature of a joint is calculated as follows:

$$\textbf{J}_k = \textbf{Pooling}(\textbf{MLP}_{vis}(v_k),\textbf{MLP}_{pos}(p_k),\textbf{MLP}_{type}(c_k))$$
    
Graph structure of the connected joints of frames has two different edge types:
* Edges that connect joints in the same frame
* Edges that connect joints in consecutive frames
    
The first type of edges capture the relative movements and spatial structure of the human body while the second type model the temporal human pose dynamics. $\textit{k}$-th
    
In each layer, graph nodes are updated via message passing:
    
$$\textbf{J}_k^{l+1} = \textbf{J}_k^l + \textbf{MLP}([\textbf{J}_k^{l} || \textbf{M}(\textbf{J}^{l}_{k', k' \in \mathcal{N}_{\textbf{J}_k^l}} | \textbf{J}_k^l)])$$

where $\textbf{J}_{k}^{l}$ is the feature of the $\textit{k}$-th joint. at the $\textit{l}$-th layer. $\mathcal{N}_{\textbf{J}_{k}^{l}}$ represents the set of neighbours of the $\textit{k}$-th joint, $\textbf{M}$ message aggregating function, and $[\cdot||\cdot]$ represents the concatenation of vectors

Self-attion is also used in message passing. $\textbf{J}_{kq}$ represents the query of $\textbf{J}_{k}$ and each joint $\textbf{J}_{k'}$ is transformed into $\textbf{J}_{k'v}$ (value) and $\textbf{J}_{k'k}$ (key). The final aggregated feature can be computed as:

$$\textbf{M}(\textbf{J}_{k', k' \in \mathcal{N}_{\textbf{J}_k}} | \textbf{J}^{l}_{k}) = \sum_{k' \in \mathcal{N}_{\textbf{J}_k}} \alpha_{kk'}\textbf{J}_{k'v}, $$

where $$\alpha_{kk'} = Softmax_{k'}(\textbf{J}^{T}_{kq}\textbf{J}_{k'k})$$
    
   
## 2.2. Our interpretation 

<!--- @TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them. --->

# 3. Experiments and results

## 3.1. Experimental setup

<!--- @TODO: Describe the setup of the original paper and whether you changed any settings. --->

## 3.2. Running the code

<!--- @TODO: Explain your code & directory structure and how other people can run it. --->
Directory structure:


    ├── posetrack_data
    │   ├── annotations
    │   │   ├── test
    │   │   ├── train
    │   │   └── val
    │   └─── images
    │        ├── test
    │        ├── train
    │        └── val
    ├── gnn_images
    ├── gnn_joints
    ├── gnn_models
    ├── models
    └── poseval

First, requirements.txt should be downloaded. 
Pretrained hrnet_w32_386_288.pth model should be downloaded and placed to the main folder.
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/README.md

## 3.3. Results

<!--- @TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper. --->

# 4. Conclusion

<!--- @TODO: Discuss the paper in relation to the results in the paper and your results. --->

# 5. References

<!--- @TODO: Provide your references here. --->
Orginial paper: Yang, Y., Ren, Z., Li, H., Zhou, C., Wang, X., & Hua, G. (2021). Learning dynamics via graph neural networks for human pose estimation and tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8074-8084).

Dataset: Andriluka, M., Iqbal, U., Insafutdinov, E., Pishchulin, L., Milan, A., Gall, J., & Schiele, B. (2018). Posetrack: A benchmark for human pose estimation and tracking. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5167-5176). (link: https://posetrack.net/)

Evaluation toolkit: https://github.com/leonid-pishchulin/poseval

Hrnet model that was downloaded: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/README.md

Hrnet code that was used: https://github.com/stefanopini/simple-HRNet

# Contact

<!--- @TODO: Provide your names & email addresses and any other info with which people can contact you. --->
Burak Akgül, akgulburak01@gmail.com

Süleyman Onat Çeltik, onat.celtik27@gmail.com
