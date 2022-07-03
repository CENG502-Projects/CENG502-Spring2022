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
The authors of the paper apply multiscale area attention which allows the model to attend at multiple scales and granularities and tp o learn the most appropriate level of details. They offer multsicale area attention because under conventional attention, the model only uses a preset granularity as the basic unit for calculation. They designed an attention-based convolutional neural network with VTLP data augmentation because of the limited amount of training data in IEMOCAP
# 2. The method and my interpretation

## 2.1. The original method

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

@TODO: Provide your names & email addresses and any other info with which people can contact you.
