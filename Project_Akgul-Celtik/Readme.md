# Learning Dynamics via Graph Neural Networks for Human Pose Estimation and Tracking

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
<p align-items="center">
   <img src="https://user-images.githubusercontent.com/64609605/177032733-06c31927-73f5-4117-8e70-e7542c5c6e44.png" alt="Figure 1. Overall pipeline of the proposed method">
  <p>Figure 1. Overall pipeline of the proposed method</p>
</p>

The proposed method's overall pipeline is shown in the figure above. The model uses two separate processes to find human poses in current frame _t_ of a video. In the first process, it tries to find the poses as in the classical approaches by using a Single-Frame Pose Estimation model with possible missed human detections. In the second process, it tries to make us of the previous _n_ (default value is 3 in the paper) frames to understand the motion of the people to predict the poses that would be in the next frame using a GNN Pose Prediction model. Then the result of the Single-Frame Pose Estimation and GNN Pose Prediction model are merged to obtain final poses.

A- Single-Frame Pose Estimation

Pose estimation is performed for each frame. Each human detection in a frame is cropped and then rescaled to 384×288. These detections are fed into the HRNet (pretrained on COCO) and it produces 96x72 heatmaps for each 17 joints. However, we ignore the ear joints due to the fact that paper states it uses 15 joints in 2017 dataset format. The positions of the $\textit{k}$-th joints are computed as:


$$l_{k}^{\*} = \underset{(i,j)}{\arg\max} \textbf{H}\_{ijk}$$
   
where $l_{k}^{\*}$ is the position within heatmap.
   
The training loss of the single-frame pose estimation model uses these heatmaps. The ground truth heatmaps are generated using the following 2D Gaussian distribution:

$$\textbf{H}\_{ijk}^{gt} = exp(-\frac{||(i,j)-l_{k}||^{2}\_{2}}{\sigma^{2}})$$

   
$\sigma$ is set to 3 and the model is trained with the following loss:

$$\mathcal{L}\_{e} = \sum_i^H\sum_j^W\sum_k^K ||\textbf{H}\_{ijk}^{pred} - \textbf{H}\_{ijk}^{gt} ||^{2}\_{2}$$

where $H$, $W$, $K$ are the height, the width of heatmaps, and the number of joints, respectively.

B- Dynamics Modeling via GNN

The people in history frames and current frame are used on the training of GNN model. It takes the joints in all frames as nodes and creates corresponding edges within a frame and between consecutive frames. This structure aims to understand the motion dynamics, and GNN updates the node features with respect to these dynamics. For pose prediction, use the results of the current frame.

<p align="center">
   <img width="500" src="https://user-images.githubusercontent.com/64609605/177046110-850b93dd-ac41-4163-b269-d21815cace99.png" alt="Figure 2. GNN Model">
  <p>Figure 2. GNN Model</p>
</p>

As mentioned, joints of the history frames and current frame are used as the nodes of the GNN. Three kinds of cues are used for each joint:
    
* Visual feature ($v_k$): The visual features from the backbone CNN of the single-frame pose estimator    
* Joint encoding ($c_k$): The encoding of its joint type with a learnable lookup
table    
* Position feature ($p_k$): 2D position and confidence score from pose estimator
    
All the 2D positions used fır the $p_k$ are normalized with respect to the center of last tracked pose. All of these cues are fed into different MLPs that do not share weights to have the same dimension and are merged using average pooling. Therefore, the final feature of a joint is calculated as follows:

$$\textbf{J}\_k = \textbf{Pooling}(\textbf{MLP}\_{vis}(v_k),\textbf{MLP}\_{pos}(p_k),\textbf{MLP}\_{type}(c_k))$$
    
Graph structure of the connected joints of frames has two different edge types:
* Edges that connect joints in the same frame
* Edges that connect joints in consecutive frames
    
The first type of edges capture the relative movements and spatial structure of the human body while the second type model the temporal human pose dynamics.
    
In each layer, graph nodes are updated via message passing:
    
$$\textbf{J}\_k^{l+1} = \textbf{J}\_k^l + \textbf{MLP}(\[\textbf{J}\_k^{l} || \textbf{M}(\textbf{J}^{l}\_{k', k' \in \mathcal{N}\_{\textbf{J}\_k^l}} | \textbf{J}\_k^l)])$$

where  $\textbf{J}\_k^l $ is the feature of the k-th joint at the l-th layer.  $\mathcal{N}\_{\textbf{J}_{k}^{l}} $ represents the set of neighbours of the k-th joint,  $\textbf{M} $ message aggregating function, and  $[\cdot||\cdot]$ represents the concatenation of vectors.

Self-attion is also used in message passing.  $\textbf{J}\_{kq} $ represents the query of  $\textbf{J}\_{k} $ and each joint  $\textbf{J}\_{k'} $ is transformed into  $\textbf{J}\_{k'v} $ (value) and  $\textbf{J}\_{k'k} $ (key). The final aggregated feature can be computed as:

$$\textbf{M}(\textbf{J}\_{k', k' \in \mathcal{N}\_{\textbf{J}\_k}} | \textbf{J}^{l}\_{k}) = \sum_{k' \in \mathcal{N}\_{\textbf{J}\_k}} \alpha_{kk'}\textbf{J}\_{k'v}, $$

where $$\alpha_{kk'} = Softmax_{k'}(\textbf{J}^{T}\_{kq}\textbf{J}\_{k'k})$$


 $\textbf{J}^T$ represents the transpose of  $\textbf{J} $ and the similarity is calculated as dot product of keys and queries. Then, the results are fed into the softmax function to get attention coefficients  $\alpha_{kk'}$.

Because different edges represents different meanings, two separate MLPs are used in the message aggregating function which are  $\textbf{MLP}\_{spatial}(\cdot) $ for in-frame edges, and   $\textbf{MLP}\_{temporal}(\cdot) $ for inter-frame edges.

For the pose prediction, the predictions of the last history frame is used as the predictions of joints for the current frame. Joint features obtained from the GNN prediction are fed into another prediction MLP:

$$ Prob = \textbf{MLP}\_{pred}(J)$$



## 2.2. Our interpretation 

<!--- @TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them. --->

- Paper uses MLP models on some of the inputs ($v_k$, $p_k$, $c_k$) of the GNN model. In the paper, it is not stated what the architectures of these MLP models were, so we had to assume a random value of 500 for channels. We see that the higher value performed better than smaller ones; however we couldn't do a cross validation. Moreover, GNN model's number of hidden layers were not specified, so we took it as 500 dimensions, too.

- It was stated that they used learnable lookup table in the paper, and we used the learnable ```Embedding()``` function of PyTorch on joint numbers to generate the embeddings of them. Bert paper was referenced for the embeddings, but it was not referenced exactly that they have used it.

- Average Pooling was used but the kernel size and stride of it was not given. So, we performed a similar operation by summing and taking average without using ```AvgPool1d``` which is the average pooling function of PyTorch.

- Inputs were concatenated from the three stages of the Single-Frame Pose Estimator, resulting in a 144-channel output. We took only the last channel dimension, as we could not find how we can reduce 144-channel output to 17 channel. It was stated that average pooling was used on the inputs of the model, but as it was not clearly stated how they used it we could not implement it.

- Prob from the $\textbf{MLP}\_{pred}(J)$ denotes the probability distribution over all joint types of the input node in the paper. From this statement we understood that the predicted node features are fed into an MLP to obtain positions on the image. Hence, we implemented as this way.

- The paper states that they used the Hungarian matching to to compute an one-to-one mapping between the predicted poses and the estimated poses. In order to reduce the complexity, we used the distances between centers of the bounding boxes as the similarity score.


# 3. Experiments and results

## 3.1. Experimental setup

<!--- @TODO: Describe the setup of the original paper and whether you changed any settings. --->

The original paper fine-tuned object detector on PoseTrack 2017 and PoseTrack 2018 datasets. We directly used pretrained object detector.

The authors also used PoseTrack 2017 and PoseTrack 2018 datasets to train the GNN model. We were able to only use a small portion of the PoseTrack 2018 dataset, nearly 1/10 of it.

Soft-NMS was used to incorporate pose information for discarding unwanted detections. We did not use it.

Paper trained the model by using learning rate 0.0001 for 10 epochs, and then reduced the learning rate by a factor of 10 at 5th and 8th epochs, and total training was 20 epochs. The initial learning rate of our implementation is the same but we have trained the model 3 epochs because our dataset size for training that was used was small.

## 3.2. Running the code

<!--- @TODO: Explain your code & directory structure and how other people can run it. --->
Directory structure:

    ├── posetrack_data *
    │   ├── annotations
    │   │   ├── test
    │   │   ├── train
    │   │   └── val
    │   └─── images
    │        ├── test
    │        ├── train
    │        └── val
    ├── poseval *
    ├── models *
    ├── gnn_images ~
    ├── gnn_joints ~
    ├── gnn_models ~
    ├── results ~
    ├── posetrack_train.py
    ├── posetrack_test.py
    ├── pose_hrnet_w32_384x288.pth *
    └── requirements.txt

- Folders with * on their right are the folders that should be downloaded, folders with ~ on their right are the folders that should be downloaded
- 'posetrack_data' folder should be donwloaded from [2] and added to the main directory
- 'poseval' folder should be donwloaded from [3] and added to the main directory
- 'pose_hrnet_w32_384x288.pth' file should be donwloaded from [4] (models-> pytorch -> pose_coco -> pose_hrnet_w32_384x288.pth) and added to the main directory
- 'models' folder should be donwloaded from [5] and added to the main directory

First of all, the modules that are listed in the requirements.txt should be downloaded. Pretrained ```hrnet_w32_386_288.pth``` model should be downloaded and placed to the main folder.
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/README.md

Training the GNN part is done with the ```posetrack_train.py``` file. Usage is like this:

```python posetrack_train.py --is_train <True|False> --mode {train/val/test}```: When ```--is_train``` is not used, preprocess of the images is done. This setting should be called first before beginning the training. After preprocessing step is done, code will be called with the ```--is_train``` option, so that GNN model would begin training. ```--mode``` flag uses which annotation folder would be selected.

```python posetrack_test.py --is_preprocess <True|False> --mode {train/val/test} --saved_model_path```: ```--is_preprocess``` flag should be called first to preprocess the outputs of the detection and hrnet models, so that our pretrained GNN model can generate COCO formatted results. ```--is_preprocess``` should be called at least once for that reason. ```--mode``` flag uses which annotation folder would be selected. ```--saved_model_path``` should be model's name that has been placed in the gnn_models folder.

## 3.3. Results

<!--- @TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper. --->

Although our implementation was finished, we couldn’t test it due to the lack of information about the output format. We tried to reverse engineer to understand what format we should produce; however, it was not evident in both the dataset and the paper. We also struggled with the dataset size. Even though we made some predictions that are in the wrong format, we were not able to achieve meaningful training and testing processes in a reasonable time meaning that everyone can't reproduce the paper without having high-end machines and longer times. What we got for the implementation of only the GNN test score shown below.

![image](https://user-images.githubusercontent.com/64609605/177098824-b2eaad58-77ce-4b59-bf8a-c4fab5e95890.png)
Mean Average Precision (AP) metric of our implementation on pure GNN result

The rest of the tables are scores from the paper.

![image](https://user-images.githubusercontent.com/64609605/177093044-cfc84f35-96f1-4e5d-b9d4-7481dbfecb8e.png)

![image](https://user-images.githubusercontent.com/64609605/177093078-ee543e94-c584-4f35-8cf8-8d08f667a1cb.png)

![image](https://user-images.githubusercontent.com/64609605/177093139-49ba7756-b4ac-476e-97fd-0a9713b33a4c.png)


# 4. Conclusion

<!--- @TODO: Discuss the paper in relation to the results in the paper and your results. --->
The paper proposes a new approach for both human pose estimation and tracking. The model tries to combine GNN and CNN models in order to achieve more consistent results. The GNN part is used to model the temporal dynamics of the human poses while also capturing the missed poses. When combining GNN with the human pose estimation model, it surpasses the performances of state-of-the-art models in PoseTrack 2017 and 2018 datasets

# 5. References

<!--- @TODO: Provide your references here. --->
1. Orginial paper: Yang, Y., Ren, Z., Li, H., Zhou, C., Wang, X., & Hua, G. (2021). Learning dynamics via graph neural networks for human pose estimation and tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8074-8084).

2. Dataset: Andriluka, M., Iqbal, U., Insafutdinov, E., Pishchulin, L., Milan, A., Gall, J., & Schiele, B. (2018). Posetrack: A benchmark for human pose estimation and tracking. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5167-5176). (link: https://posetrack.net/)

3. Evaluation toolkit: https://github.com/leonid-pishchulin/poseval

4. HRNet model that was downloaded: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/README.md

   4.1. Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). Deep high-resolution representation learning for human pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5693-5703).

   4.2 Xiao, B., Wu, H., & Wei, Y. (2018). Simple baselines for human pose estimation and tracking. In Proceedings of the European conference on computer vision (ECCV) (pp. 466-481).
   
   4.3 Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., ... & Xiao, B. (2020). Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 43(10), 3349-3364.

5. HRNet code that was used: https://github.com/stefanopini/simple-HRNet

# Contact

<!--- @TODO: Provide your names & email addresses and any other info with which people can contact you. --->
- Burak Akgül, akgulburak01@gmail.com

- Süleyman Onat Çeltik, onat.celtik27@gmail.com
