# Speech Emotion Recognition with Multiscale Area Attention  and Data Augmentation 

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction
This paper was presented in the International Conference on Acoustics, Speech, & Signal Processing2021 (ICASSP) in 2021. In this paper, authors propose a method that they can apply **multiscale area attention** in a deep convolutional neural network to attend emotional characteristics with varied granularities. The main advantange of this paper is the classifier can benefit from an ensemble of attentions with different scales.
  - To deal with the data sparsity, they conduct data augmentation with **vocal tract length perturbation (VTLP)** to improve the generalization capability of the classifer.
  - **Index Terms:** Speech Emotion Recognition, convolutional neural network, attention mechanism and data augmentation.
  - I will try to reproduce the results given in the paper according to the methods described in the paper.
  - Differences in results together with the differences in implementations are given and briefly discussed in the following sections.

## 1.1. Paper summary


   The authors of the paper apply multiscale area attention which allows the model to attend at multiple scales and granularities and to learn the most appropriate level of details. They offer multsicale area attention because under conventional attention, the model only uses a preset granularity as the basic unit for calculation. They designed an attention-based convolutional neural network with VTLP data augmentation because of the limited amount of training data in IEMOCAP. The advantage of the proposed method is tp be able to dynamically adaption to different areas of interest. Tradional methods uses typical attention neural network classifiers with CNN and LSTM’s. 
   We can summarize the paper main contributions as follows:
   <br/>
    - First attempt for applying **multiscale area attention** to SER,
    <br/>
    - Data augmentation on the IEMOCAP dataset with **vocal tract length perturbation(VTLP)** to able to achieve accuracy improvement


# 2. The method and my interpretation

## 2.1. The original method

### 2.1.1 Model Architecture and Details
![image](https://user-images.githubusercontent.com/53267971/177052946-75265b17-0787-49ff-b830-a1dae08fa20e.png)
 <br/>
**Figure 1**: The architecture of the CNN with attention used as a classifier in this work
   <br/>
   - **Librosa** for logMel spectrogram(Time-frequency information in Mel scale, which human can interpret) features.
   - Two **parallel convolutional layers** to extract textures from
      - Time axis,
      - Frequency axis
   - **Four consequtive convolutional layers** with Batch Normalization
   - **Area attention layer** followed by fully connected layer for classification.
    <br/>
### 2.1.2 Multiscale Area Attention

![image](https://user-images.githubusercontent.com/53267971/177053093-90e325a5-41a5-4f4f-81a4-839a4e59556b.png)
 <br/>
**Figure 2:** Multiscale Area Attention 
  - Attend at multiple scales in order to calculate attention in units of areas
  - Define two important attention paramters, **key** and **value** accordingly.
  - In the original paper, they use
    - **For Key** : Max, Mean and Sample(adding a perturbation during training)
    - **For Value** : Max, Mean and Sum
### 2.1.3 Data Augmentation

- Data augmentation is done with 
**Vocal tract length perturbation(VTLP)** the details are **NOT explained** in the paper. 

## 2.2. Our interpretation 
### 2.2.1 Model Architecture and Details
 
  - This is exactly same since it is clearly given in the paper. For details, please look at the above section **2.1.1**
  - **Dataset** : Interactive Emotional Dyadic Motion Capture(IEMOCAP) with paper used:
  - You can submit for your request to be able to download the dataset :
      - https://sail.usc.edu/iemocap/release_form.php
  - 9 types of emotions with imbalances
    - Paper used: ‘ Neutral’, ‘Sad’, ‘Angry’ and ‘Happy’
    - My implementation :‘ Neutral’, ‘Sad’, ‘Angry’, ‘Happy’, ‘Frustration’, ‘Excitement’

  
### 2.2.2 Multiscale Area Attention
  - My implementation is based on **mean of an area as key**, and **sum of an area as a value** since with these values they can be evaluated in a similar to ordinary attention. 
### 2.2.3 Evaluation Metrics
  - Weighted Accuracy (WA)
  - Unweighted Accuracy (UA)
  - ACC - average of (WA) and (UA)
    - In this paper, our success will based on the ACC.
    
### 2.2.4 Data Augmentation
![image](https://user-images.githubusercontent.com/53267971/177053613-593e2a8e-2d76-4d18-aa18-4dbdc7314619.png)
<br/>
**Figure 3:** Vocal tract length perturbation(VTLP) of frequency expansion with warp factor
- Not explained in the paper, but given in the paper  ‘’ VTLP improves speech recognition’’ by researches from Universities Toronto.
- It simply means that the frequency axis of the spectrogram of each speaker is linearly warped using a warp factor.
- Aim is randomly generating a random warp factor.
- Implementation is very easy due to **nlpaug** library for textual augmentation in machine learning. 
  - nlpaug.augmenter.audio.vtlp
  - Since details are not given in the paper, I used the **default parameters** of the function vtlp with the parameters:
  - nlpaug.augmenter.audio.vtlp.VtlpAug(sampling_rate, zone=(0.2, 0.8), coverage=0.1, fhi=4800, factor=(0.9, 1.1), name='Vtlp_Aug', verbose=0, stateless=True) 

# 3. Experiments and results

## 3.1. Experimental setup

  - Divide dataset into a training set(80 %) and test test(20 %)
  - 5-fold cross validation
  - Each utterance is divided into **2-second segment**.
    - **1-second** (for training) or **1.5-second** (for testing) overlap between segments.
  -  **Librosa** for logMel spectrogram(Time-frequency information in Mel scale, which human can interpret) features.
   - Two **parallel convolutional layers** to extract textures from
      - Time axis,
      - Frequency axis
   - **Four consequtive convolutional layers** with Batch Normalization
   - **Area attention layer** followed by fully connected layer for classification.
    <br/>
   - All other details can be found in Model Architecture part.
## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results
- You can see the results of the paper as well as my implementation together with comparisions with details.
### 3.3.1 Selection of maximum area size
**Proposed:** According to high ACC value two maximum area size is suggested:
  - 4x4 max-area size without augmentation
  - 3x3 max-area size with augmentation.

![image](https://user-images.githubusercontent.com/53267971/177053810-f78928aa-53c4-4004-945a-6bdbcc80ee38.png)
<br/>
**Figure 4**: Three evaluation metrics versus accuracy plots in the paper with/without data augmentation  
### 3.3.2 My result compared to paper 
![image](https://user-images.githubusercontent.com/53267971/177054598-bc02efcb-daf0-46e3-95fb-4665a1c9c4a0.png)
<br/>
**Figure 5**: My implementation with three evaluation metrics versus accuracy plots in the paper **only with** data augmentation. 
<br/>

- I obtained the **max-area size 2x2**(In the paper there was 3x3) with average accuracy as **59.73** % while paper propeses max-area size 3x3 with average accuracy as **78.44 %. The reason for this reduction may due to for the following reasons
  - I have used extra 2 emotions in total 6 emotions instead of 4 emotions, where the authors of the paper take one class ‘’ happy’’ and ‘’excitement’’ to increase the amount of data.
  - Augmentation procedure is not explained in detail, and I used the ‘’default’’ parameters of the nlpaug library.
  
  ![image](https://user-images.githubusercontent.com/53267971/177054691-2da45904-8dbf-498b-bbce-cacaaf8b4ba1.png)
  <br/>
**Figure 6**: Comparision of avarage accuracy values with the original paper and my implementation for the different choice of key and value parameters.
 - My implementation is based on **mean of an area as key**, and **sum of an area as a value** since with these values they can be evaluated in a similar to ordinary attention. 
 - In the original paper, they use
    - **For Key** : Max, Mean and Sample(adding a perturbation during training)
    - **For Value** : Max, Mean and Sum
 - **COMMENT**: The reason with the sample key is high ACC value is that by doing so we are introducing greater randomness to the training procedure.

# 4. Conclusion
As I mentioned above, the reason why I obtain different result can be summarized below.
  - Use of the extra 2 emotions in total 6 emotions instead of 4 emotions, where the authors of the paper take one class ‘’ happy’’ and ‘’excitement’’ to increase the amount of data. But I believe that these two emotions are not highly correlated since excitement does not always mean people who are happy, or vice versa. Therefore, I treat them with two different classes although due to reduction in the amount of data my performance will be reduced.(But I think it is still more realistic.) 
  - Augmentation procedure is not explained in detail, and I used the ‘’default’’ parameters of the **nlpaug** library.
  - For training part, I use the parameters of the paper **Head Fusion Net** which is the paper of the same authors published before that paper, where they did **NOT** use the **multi-scale area attention**. Since I have no any information about the training procedure of the paper, I used them. This paper repo can be also found in the references.
  - For me the **pros** of the paper can be listed as follows:
    - A good application of Multiscale Area Attention.
    - Architecture of the overall process are well explained and easily implemented in the paper including attention layer.
    - Experimental results are presented in details with different parameters.
  - Together with the **cons** :
    - IEMOCAP dataset is imbalanced with 9 emotions. 
    - It requires permission to use. => Difficult to explore the similar approaches in the research.  
    - Data Augmentation with VTLP is not explained in the paper. Since it has several parameters, results can be different.
    - Not possible to predict how the paper will be developed in further studies.
    - Training parameters are not given in the paper, which is set with the help of **[3]** 

# 5. References
[1] Mingke Xu, Fan Zhang, Xiadong Cui, Wei Zhang ‘’Speech Emotion Recognition with Multiscale Area Attention  and Data Augmentation’’, ICASSP 2021
<br/>
[2] Navdeep Jaitly and Geoffrey E.Hinton ‘’ Vocal tract length perturbation improves speech recognition’’ in Proc. ICML Workshop on Deep Learning for Audio, Speech and Language, 2013, vol 117.
<br/>
[3] For the model and training details : https://github.com/lessonxmk/head_fusion (The paper of the same authors published before that paper.)

# Contact

Yalçın Cenik, email: yalcin.cenik@metu.edu.tr
