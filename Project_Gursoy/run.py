import argparse
import os
import numpy as np
import torch
import torchvision
from imgaug import augmenters as iaa
import tensorflow.compat.v1 as tf
import os.path as osp
from transform import TPS_SpatialTransformerNetwork
from feature_extractor import ResNet_FeatureExtractor
from sequence_model import BidirectionalLSTM
from prediction import Attention

LARGE_NUM = 1e9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def apply_stochastic_data_augmentation(data):
    aug = iaa.Sequential([iaa.SomeOf((1, 5),
                            [
                                iaa.LinearContrast((0.5, 1.0)),
                                iaa.GaussianBlur((0.5, 1.5)),
                                iaa.Crop(percent=((0, 0.4),(0, 0),(0, 0.4),(0, 0.0)), keep_size = True),
                                iaa.Crop(percent=((0, 0.0), (0, 0.02), (0, 0), (0, 0.02)), keep_size = True),
                                iaa.Sharpen(alpha=(0.0, 0.5), lightness = (0.0, 0.5)),
                                iaa.PiecewiseAffine(scale=(0.02, 0.03), mode ='edge'),
                                iaa.PerspectiveTransform(scale = (0.01, 0.02)),], random_order = True)])
    return torch.from_numpy(aug(images=np.array(data)))

def apply_basic_data_augmentation(data):
    aug = iaa.Sequential([iaa.SomeOf((1, 5),
                            [
                                iaa.LinearContrast((0.5, 1.0)),
                                iaa.GaussianBlur((0.5, 1.5)),
                                iaa.Crop(percent=((0, 0.4),(0, 0),(0, 0.4),(0, 0.0)), keep_size = True),
                                iaa.Crop(percent=((0, 0.0), (0, 0.02), (0, 0), (0, 0.02)), keep_size = True)],
                                random_order = True)])
    return torch.from_numpy(aug(images=np.array(data)))                            

class SeqCLR(torch.nn.Module):
    def __init__(self, num_classes, batch_size, train_iter_num, train_dec_iter_num, 
                pretrained_encoder_model_path, pretrained_decoder_model_path,
                prediction, instanceMapping, projection=None, sequenceModel=None):
        super(SeqCLR, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.train_iter_num = train_iter_num
        self.train_dec_iter_num = train_dec_iter_num
        self.pretrained_encoder_model_path = pretrained_encoder_model_path
        self.pretrained_decoder_model_path = pretrained_decoder_model_path
        self.transformation = self.init_transformation()
        self.featureExtractor = self.init_featureExtractor()
        self.sequenceModel = self.init_sequenceModel(sequenceModel)
        self.prediction = self.init_prediction(prediction)
        self.projectionHead = self.init_projectionHead(projection)
        self.instanceMapping = self.init_instanceMapping(instanceMapping)

    def init_transformation(self):
        return TPS_SpatialTransformerNetwork(F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=3).to(DEVICE)

    def init_featureExtractor(self):
        return ResNet_FeatureExtractor(3, 512).to(DEVICE)

    def init_sequenceModel(self, sequenceModel):
        if sequenceModel:
            return torch.nn.Sequential(
                BidirectionalLSTM(512, 256, 256),
                BidirectionalLSTM(256, 256, 256)).to(DEVICE)
        else:
            return None

    def init_prediction(self, prediction):
        if prediction == 'CTC':
            return torch.nn.Linear(256, self.num_classes).to(DEVICE)
        elif prediction == 'Attn':
            return Attention(256, 256, self.num_classes).to(DEVICE)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def init_projectionHead(self, projection):
        return None

    def init_instanceMapping(self, instanceMapping):
        if instanceMapping == 'WindowToInstance':
            return torch.nn.AdaptiveAvgPool2d((self.batch_size, 1)).to(DEVICE)
        elif instanceMapping == 'AllToInstance':
            return torch.nn.AvgPool2d((self.batch_size, 1), stride=5).to(DEVICE)
        elif instanceMapping == 'FrameToInstance':
            return torch.nn.Identity().to(DEVICE)
        else:
            raise Exception('Unknown Instance Mapping')
    
    def contrastive_loss(self, hidden1, hidden2):
        hidden1 = hidden1.view(-1, self.batch_size)
        hidden2 = hidden2.view(-1, self.batch_size)
        hidden1 = hidden1.cpu().detach().numpy()
        hidden2 = hidden2.cpu().detach().numpy()

        temperature = 1.0
        labels = tf.one_hot(tf.range(self.batch_size), self.batch_size * 2)
        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature

        loss_a = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ab, logits_aa], 1), weights=1.0)
        loss_b = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ba, logits_bb], 1), weights=1.0)
        loss = loss_a + loss_b
        return loss, logits_ab, labels

    def decoder_loss(self, preds, labels):
        loss = torch.nn.NLLLoss().to(DEVICE)
        return loss(preds, labels)

    def train(self, train_data):
        self.optimizer = torch.optim.Adadelta(self.parameters(),
                                            weight_decay=0.0001, 
                                            lr=10,
                                            rho=0.95)
        iter_num = 0
        while iter_num < self.train_iter_num:
            print("iter_num: ", iter_num)
            for batch in train_data:
                images, labels = batch
                images2 = apply_stochastic_data_augmentation(images)
                images, images2 = images.to(DEVICE), images2.to(DEVICE)
                if self.transformation:
                    images = self.transformation(images).to(DEVICE)
                    images2 = self.transformation(images2).to(DEVICE)
                features = self.featureExtractor(images).to(DEVICE)
                features2 = self.featureExtractor(images2).to(DEVICE)
                adp = torch.nn.AdaptiveAvgPool2d((None, 1))
                features = adp(features.permute(0, 3, 1, 2))
                features = features.squeeze(3).to(DEVICE)
                features2 = adp(features2.permute(0, 3, 1, 2))
                features2 = features2.squeeze(3).to(DEVICE)
                if self.sequenceModel:
                    features = self.sequenceModel(features).to(DEVICE)
                    features2 = self.sequenceModel(features2).to(DEVICE)
                if self.projectionHead:
                    features = self.projectionHead(features).to(DEVICE)
                    features2 = self.projectionHead(features2).to(DEVICE)
                instances = self.instanceMapping(features).to(DEVICE)
                instances2 = self.instanceMapping(features2).to(DEVICE)
                loss, logits, labels = self.contrastive_loss(instances, instances2)
                self.optimizer.zero_grad()
                loss.backward()
            torch.save(self, self.pretrained_encoder_model_path)
            iter_num += 1
            if iter_num == self.train_iter_num*0.6 or iter_num == self.train_iter_num*0.8:
                self.optimizer.param_groups[0]['lr']  *= 0.1

    def train_decoder(self, train_data):
        model = torch.load(self.pretrained_encoder_model_path)
        iter_num = 0
        while iter_num < self.train_dec_iter_num:
            for batch in train_data:
                images, labels = batch
                images = apply_basic_data_augmentation(images)
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                if model.transformation:
                    images = model.transformation(images).to(DEVICE)
                features = model.featureExtractor(images).to(DEVICE)
                adp = torch.nn.AdaptiveAvgPool2d((None, 1))
                features = adp(features.permute(0, 3, 1, 2))
                features = features.squeeze(3).to(DEVICE)
                if model.sequenceModel:
                    features = model.sequenceModel(features).to(DEVICE)
                preds =  model.prediction(features).to(DEVICE)
                loss = model.decoder_loss(preds[:, -1], labels).to(DEVICE)
                model.optimizer.zero_grad()
                loss.backward()
            torch.save(model, self.pretrained_decoder_model_path)
            iter_num += 1
            if iter_num == self.train_dec_iter_num*0.6 or iter_num == self.train_dec_iter_num*0.8:
                model.optimizer.param_groups[0]['lr']  *= 0.1

    def test(self, test_data):
        pretrained_decoder_model = torch.load(self.pretrained_decoder_model_path)
        preds = []
        labels= []
        for t in test_data:
            image, label = t
            image = apply_basic_data_augmentation(image)
            pred = pretrained_decoder_model.prediction(image)
            preds.append(pred)
            labels.append(label)
        print("Accuracy: ", self.accuracy(preds, labels)[0])

    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_path', default='./train', help='path to train dataset')
    parser.add_argument('-test_data_path', default='./test', help='path to test dataset')
    parser.add_argument('-batch_size', default=16, type=int, help='batch size for training')
    parser.add_argument('-train_iter_num', default=3000, type=int, help='iteration num for train encoder')
    parser.add_argument('-train_dec_iter_num', default=500, type=int, help='iteration num for train decoder')
    parser.add_argument('-pretrained_encoder_model_path', default='./encoder.pth', help='path to save/load trained encoder')    
    parser.add_argument('-pretrained_decoder_model_path', default='./decoder.pth', help='path to save/load trained decoder')
    parser.add_argument('-prediction', default="CTC", help='prediction type')
    parser.add_argument('-ins_map', default="WindowToInstance", help='instance mapping type')
    parser.add_argument('-projection', default=None, help='projection type')
    parser.add_argument('-seq_model', default=True, help='sequence modeling type')
    args = parser.parse_args()

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    num_classes = len(os.listdir(train_data_path))

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(32,100)), torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, prefetch_factor=2,
            persistent_workers=False)

    seqclr = SeqCLR(num_classes, args.batch_size, args.train_iter_num, args.train_dec_iter_num,
                    pretrained_encoder_model_path=args.pretrained_encoder_model_path, pretrained_decoder_model_path=args.pretrained_decoder_model_path,
                    prediction=args.prediction, instanceMapping=args.ins_map, projection=args.projection, sequenceModel=args.seq_model).to(DEVICE)
    seqclr.train(train_data)
    seqclr.train_decoder(train_data)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(32,100)), torchvision.transforms.ToTensor()])
    test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=transform)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=1,
            shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, prefetch_factor=2,
            persistent_workers=False)
    seqclr.test(test_data)

if __name__ == "__main__":
    main()
