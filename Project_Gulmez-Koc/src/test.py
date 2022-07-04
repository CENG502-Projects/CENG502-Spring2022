from configuration import *
from VocDataset import *
from ClassAwareSampler import *
from Network import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def F1_score(prob, label):
    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))
    #print("Acc:{} Precision:{} Recall:{}".format(accuracy,precision,recall))
    F1scr = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, F1scr

# Initials
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
test_data = VOC(root=test_dataset_output_path, imgtransform=transform_train)
testLoader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
net = torch.load(test_weight_path)
net = net.to(device)
net.eval()
  
for testData,testLabel in testLoader:
    uniform , resampled = net(testData)
    prediction_raw = ( uniform + resampled )/ 2
    prediction = (prediction_raw>threshold).float()
    precision, recall , f1 = F1_score(testLabel, prediction)

