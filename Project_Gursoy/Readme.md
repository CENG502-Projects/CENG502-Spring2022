# Paper title 
Sequence-to-Sequence Contrastive Learning for Text Recognition

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction
In this paper, sequence-to-sequence contrastive learning model architecture is proposed for text recognition task. It is published in CVPR 2021. In this project, the implementation of this model has been tried.

## 1.1. Paper summary
In this paper, in order to use contrastive learning method for text recognition task, sequence-to-sequence architecture is used. Since many contrastive learning methods are designed for object detection or object classification tasks, they consider images as atomic elements. However, texts have sequential architecture with characters, so they consider that modeling them as a sequence of image frames is more suitable. Therefore, they implement sequence-to-sequence modeling with contrastive learning for this task. 

# 2. The method and my interpretation

## 2.1. The original method

Firstly, an encoder model is trained with unlabeled data. Contrastive learning part begins with data augmentation and augmented version of same images is processed by encoder, projection head and instance mapping. Then contrastive loss is calculated. In augmentation, they use methods like vertical cropping, blurring, random noise etc. Then, a decoder is trained by finetuning the trained encoder using labeled data. In transformation part, they normalize the input text image using Thin Plate Spline transformation network. Then, these normalized images are given to feature extraction part consisting of ResNet. Then, Bidirectional LSTM part captures the contextual information from these features. And a text decoder predicts the text. They use two options for decoder, which are CTC and Attention. In the model, projection head is optional, but if it is used, Multilayer perceptron and BiLSTM options exist. In instance-mapping function, they apply three different methods which are all-to-instance which is average pooling, window-to-instance which is adaptive average pooling and frame-to-instance which is idendity operation. Then, they use contrastive loss function. 

## 2.2. Our interpretation 
Since I did not sure the implementation details of projection head, I discarded this part.

# 3. Experiments and results

## 3.1. Experimental setup
In experiments, they use 6 different datasets, however I just try the model with IAM dataset. This dataset consists of about 115000 English handwritten word images. But, I eliminated classes less than 50 images to reduce the size of dataset. For evaluation, they first train encoder part with self supervised methods using unlabeled data. Then, they freeze the encoder weights and train on top of it a CTC or attention decoder with all the labeled data. In implementation, they use 300000 iterations for encoder training and 50000 iterations for decoder training. But, I trained the encoder with 3000 iterations and trained the decoder with only 500 iterations. They use batch size as 256 and I used 16.

## 3.2. Running the code
train and test data folders set must be in /class/images format
python run.py -train_data_path -test_data_path -batch_size -train_iter_num -train_dec_iter_num -pretrained_encoder_model_path -pretrained_decoder_model_path -prediction -ins_map -projection -seq_model

## 3.3. Results

# 4. Conclusion

To summarize, the key contributions of this work are:
- A contrastive learning approach is used with sequence to sequence recognition.
- Instead of taking text images completely, they extract feature maps as a sequence of individual instances. And they used contrastive learning in a sub-word level, such that each image yields several positive pairs and multiple negative examples.
- They paid attention to data augmentation techniques for text images.
- They achieve high performance on handwritten text.

# 5. References
Aberdam, A., Litman, R., Tsiper, S., Anschel, O., Slossberg, R., Mazor, S., Manmatha, R., Perona, P.: Sequence-to-sequence contrastive learning for text recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15302–15312 (2021)
https://github.com/google-research/simclr
https://github.com/sthalles/SimCLR/
https://github.com/clovaai/deep-text-recognition-benchmark

# Contact
Ceren Gürsoy - ceren.gursoy@metu.edu.tr
