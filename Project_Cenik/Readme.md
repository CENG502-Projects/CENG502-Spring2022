# Speech Emotion Recognition with Multiscale Area Attention  and Data Augmentation 

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction
This paper was presented in the International Conference on Acoustics, Speech, & Signal Processing2021 (ICASSP) in 2021. In this paper, authors propose a method that they can apply **multiscale area attention** in a deep convolutional neural network to attend emotional characteristics with varied granularities. The main advantange of this paper is the classifier can benefit from an ensemble of attentions with different scales.
  - To deal with the data sparsity, they conduct data augmentation with **vocal tract length perturbation (VTLP)** to improve the generalization capability of the classifer.
  - **Index Terms:** Speech Emotion Recognition, convolutional neural network, attention mechanism and data augmentation.
  - I will try to reproduce the results given in the paper according to the methods described in the paper.

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.
   The authors of the paper apply multiscale area attention which allows the model to attend at multiple scales and granularities and to learn the most appropriate level of details. They offer multsicale area attention because under conventional attention, the model only uses a preset granularity as the basic unit for calculation. They designed an attention-based convolutional neural network with VTLP data augmentation because of the limited amount of training data in IEMOCAP. The advantage of the proposed method is tp be able to dynamically adaption to different areas of interest. Tradional methods uses typical attention neural network classifiers with CNN and LSTM’s. 
   We can summarize the paper main contributions as follows:
   <br/>
    - First attempt for applying **multiscale area attention** to SER,
    <br/>
    - Data augmentation on the IEMOCAP dataset with **vocal tract length perturbation(VTLP)** to able to achieve accuracy improvement


# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.
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
### 2.1.2 Data Augmentation

- Data augmentation is done with 
**Vocal tract length perturbation(VTLP)** the details are **NOT explained** in the paper. 

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.
 <br/>
### 2.2.1 Model Architecture and Details
 
  - This is exactly same since it is clearly given in the paper. For details, please look at the above section **2.1.1**
  - **Dataset** : Interactive Emotional Dyadic Motion Capture(IEMOCAP) with paper used:
  - You can submit for your request to be able to download the dataset :
      - https://sail.usc.edu/iemocap/release_form.php
  - 9 types of emotions with imbalances
    - Paper used: ‘ Neural’, ‘Sad’, ‘Angry’ and ‘Happy’
    - My implementation :‘ Neural’, ‘Sad’, ‘Angry’, ‘Happy’, ‘Frustration’, ‘Excitement’

  
### 2.2.2 Multiscale Area Attention
  - My implementation is based on **mean of an area as key**, and **sum of an area as a value** since with these values they can be evaluated in a similar to ordinary attention. 
### 2.2.3 Evaluation Metrics
  - Weighted Accuracy (WA)
  - Unweighted Accuracy (UA)
  - ACC - average of (WA) and (UA)
    - In this paper, out success will based on the ACC.
    
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

@TODO: Describe the setup of the original paper and whether you changed any settings.
  - Divide dataset into a training set(80 %) and test test(20 %)
  - 5-fold cross validation
  - Each utterance is divided into **2-second segment**.
    - **1-second** (for training) or **1.5-second** (for testing) overlap between segments.
    <br/>
## 3.1.1 Selection of maximum area size
**Proposed:** According to high ACC value two maximum area size is suggested:
  - 4x4 max-area size without augmentation
  - 3x3 max-area size with augmentation.

![image](https://user-images.githubusercontent.com/53267971/177053810-f78928aa-53c4-4004-945a-6bdbcc80ee38.png)
3.1.1.1 My Result compared to paper
  
![image](https://user-images.githubusercontent.com/53267971/177053845-6a49e827-ca6f-456c-947b-dc3a18323b32.png)
- I obtained the max-area size 2x2 with average accuracy as 59.73 % while paper propeses max-area size 3x3 with average accuracy as 78.44 %. The reason for this reduction may due to for the following reasons
  - I have used extra 2 emotions in total 6 emotions instead of 4 emotions, where the authors of the paper take one class ‘’ happy’’ and ‘’excitement’’ to increase the amount of data.
  - Augmentation procedure is not explained in detail, and I used the ‘’default’’ parameters of the nlpaug librar



## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
