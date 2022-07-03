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
<p align="center">
    *Figure 1. Overall pipeline of the proposed method*
</p>

The proposed method's overall pipeline is shown in the figure above. The model uses two separate processes to find human poses in current frame _t_ of a video. In the first process, it tries to find the poses as in the classical approaches by using a Single Frame Pose Estimation model with possible missed human detections. In the second process, it tries to make us of the previous _n_ (default value is 3 in the paper) frames to understand the motion of the people to predict the poses that would be in the next frame using a GNN Pose Prediction model. Then the result of the Single Frame Pose Estimation and GNN Pose Prediction model are merged to obtain final poses.

<h2>1- Single Frame Pose Estimation<h2>


## 2.2. Our interpretation 

<!--- @TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them. --->

# 3. Experiments and results

## 3.1. Experimental setup

<!--- @TODO: Describe the setup of the original paper and whether you changed any settings. --->

## 3.2. Running the code

<!--- @TODO: Explain your code & directory structure and how other people can run it. --->
Directory structure:
└── dataset<br>
    ├── posetrack_data<br>
    │         ├── annotations<br>
    │         │    ├── test<br>
    │         │    ├── train<br>
    │         │    └── val<br>
    │         └─── images<br>
    │              ├── test<br>
    │              ├── train<br>
    │              └─── val<br>
    ├── gnn_images<br>
   ├── gnn_joints<br>
    ├── gnn_models<br>
    ├── models<br>
    └── poseval<br>

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
