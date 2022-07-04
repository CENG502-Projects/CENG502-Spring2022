from configuration import *
from VocDataset import *
from ClassAwareSampler import *
from Network import *
from apmeter import *

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from copy import deepcopy
import time

from sklearn.metrics import average_precision_score, accuracy_score, label_ranking_average_precision_score, precision_score


# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
training_data = VOC(root=lt_dataset_output_path, imgtransform=transform_train)

test_data = VOC(root=test_dataset_output_path, imgtransform=transform_train)
testLoader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


def splitDataset(training_data, validation_set_ratio):
  train_count = int(len(training_data)*(1-validation_set_ratio))
  val_count = len(training_data) - train_count
  train_set0, val_set0 = random_split(training_data, [train_count, val_count], generator=torch.Generator().manual_seed(seed))
  train_idxs = train_set0.indices
  train_set = deepcopy(training_data)
  train_set.reduceByIndexing(train_idxs)
  val_idxs = val_set0.indices
  val_set = deepcopy(training_data)
  val_set.reduceByIndexing(val_idxs)
  return train_set, val_set

train_set, val_set = splitDataset(training_data, validation_set_ratio)
if False: # quick test
  train_set, val_set = splitDataset(val_set, validation_set_ratio)
  
if debug_mode:
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        label = label.index(1)
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


net = Net().to(dev)
print(net)

uniform_dataloader = DataLoader(train_set, batch_size=batch_size_, shuffle=False)
rsampler = ClassAwareSampler(dataset=train_set,num_sample_class=len(labels_map),samples_per_gpu=batch_size_)
rebalanced_dataloader = DataLoader(train_set, batch_size=batch_size_, shuffle=False,sampler=rsampler)
optimizer = torch.optim.SGD(net.parameters(), weight_decay = weight_decay_, lr=lr_, momentum=momentum_)

uniform_dataloader_val = DataLoader(val_set, batch_size=batch_size_, shuffle=False)

Lcls = nn.BCEWithLogitsLoss()
Lcon = nn.MSELoss()

# Tensorboard
writer = SummaryWriter("{}/{}".format(tnsrbrd_dir,int(time.time())))

# https://discuss.pytorch.org/t/is-there-any-nice-pre-defined-function-to-calculate-precision-recall-and-f1-score-for-multi-class-multilabel-classification/103353
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

def runOnDataset(dataloader,name,colour="blue"):
  global net
  iterations = int(len(dataloader) / batch_size_)
  counter = 0
  precision = 0
  recall = 0
  F1scr = 0
  net.eval()
  pred_raw_all = np.ndarray([0,len(labels_map)])
  pred_all = np.ndarray([0,len(labels_map)])
  y_all = np.ndarray([0,len(labels_map)])
  for i in tqdm(range(len(dataloader)), desc=name, colour=colour):
    counter+= 1
    x, y = next(iter(dataloader))
    y_all = np.concatenate((y_all, y.cpu().detach().numpy()))

    uniform , resampled = net(x)

    prediction_raw = ( uniform + resampled )/ 2
    pred_raw_all = np.concatenate((pred_raw_all, prediction_raw.cpu().detach().numpy()))

    prediction = (prediction_raw>threshold).float()
    pred_all = np.concatenate((pred_all, prediction.cpu().detach().numpy()))

    tmp_precision, tmp_recall , tmp_f1 = F1_score(y, prediction)
    precision += tmp_precision.item()
    recall += tmp_recall.item()
    F1scr += tmp_f1.item()
    
  precision /= counter
  recall /= counter
  F1scr /= counter

  return precision, recall, F1scr, pred_raw_all, pred_all, y_all

def ltAnalysis(mAP_cls,batch_counter,name):
  head = 0
  medium = 0
  tail = 0
  for label in lt["head"]:
    id = voc_labels[label]
    head += mAP_cls[id]
  for label in lt["medium"]:
    id = voc_labels[label]
    medium += mAP_cls[id]
  for label in lt["tail"]:
    id = voc_labels[label]
    tail += mAP_cls[id]
  head = head/6
  medium = medium/6
  tail = tail/8
  writer.add_scalar("{}/head".format(name), head, batch_counter)
  writer.add_scalar("{}/medium".format(name), medium, batch_counter)
  writer.add_scalar("{}/tail".format(name), tail, batch_counter)

apm_val = APMeter()
apm_test = APMeter()

total_iterations = int(len(train_set) / batch_size_)
total_iterations_val = int(len(val_set) / batch_size_)
patience = 0
epoch_counter = 0 
batch_counter = 0
F1scr_val = 0
while patience < patience_level:
  epoch_counter += 1
  avg_loss_iters = 0
  ##########  Train #############
  net.train()
  for i in tqdm(range(len(uniform_dataloader)), desc="train", colour='green'):
      batch_counter += 1
      optimizer.zero_grad()

      xU, yU = next(iter(uniform_dataloader))
      xR, yR = next(iter(rebalanced_dataloader))

      loss = torch.tensor([0.0], device = dev)
      if uniform_branch_active:
        u , uHat = net(xU)
        loss += Lcls(u,yU)
      if resampled_branch_active:
        rHat , r = net(xR)
        loss += Lcls(r,yR)
      if uniform_branch_active and resampled_branch_active and logit_consistency:
        loss +=lambda_ * ( Lcon(u,uHat) + Lcon(r,rHat))

      avg_loss_iters += loss.item()
      loss.backward()
      optimizer.step()
      avg_loss_iters += loss.item()
      writer.add_scalar("Training/loss", loss, batch_counter)

      if uniform_branch_active:
        uniform_result = (u>threshold).float()
        uniform_precision, uniform_recall, uniform_F1scr = F1_score(yU, uniform_result)
        writer.add_scalar("Training/uniform/precision", uniform_precision, batch_counter)
        writer.add_scalar("Training/uniform/recall", uniform_recall, batch_counter)
        writer.add_scalar("Training/uniform/F1scr", uniform_F1scr, batch_counter)

      if resampled_branch_active:
        resampled_result = (r>threshold).float()
        resampled_precision, resampled_recall, resampled_F1scr = F1_score(yR, resampled_result)
        writer.add_scalar("Training/resampled/precision", resampled_precision, batch_counter)
        writer.add_scalar("Training/resampled/recall", resampled_recall, batch_counter)
        writer.add_scalar("Training/resampled/F1scr", resampled_F1scr, batch_counter)

      if batch_counter % 40 == 0:
        ##########  Validation #############
        precision_val,recall_val,F1scr_val_new, pred_raw_all_val, pred_all_val, y_all_val = runOnDataset(uniform_dataloader_val,"val")          
        writer.add_scalar("Validation/precision", precision_val, batch_counter)
        writer.add_scalar("Validation/recall", recall_val, batch_counter)
        writer.add_scalar("Validation/F1scr", F1scr_val_new, batch_counter)
        #mAP_raw_val, APs_raw_val = eval_map(pred_raw_all_val, y_all_val, avg="samples")
        #mAP_val, APs_val = eval_map(pred_all_val, y_all_val, avg="samples")
        #aa1 = average_precision_score(pred_all_val.astype(bool), np.asarray(y_all_val).astype(bool), average="samples")
        apm_val.add(torch.from_numpy(pred_all_val), torch.from_numpy(y_all_val))
        mAP_val_cls = apm_val.value().cpu().detach().numpy()
        mAP_val = mAP_val_cls.mean()
        ltAnalysis(mAP_val_cls,batch_counter,"Validation")

        writer.add_scalar("Validation/mAP", mAP_val, batch_counter)

        ##########  Test #############
        precision_test,recall_test,F1scr_test, pred_raw_all_test, pred_all_test, y_all_test = runOnDataset(testLoader,"test","red")          
        writer.add_scalar("Test/precision", precision_test, batch_counter)
        writer.add_scalar("Test/recall", recall_test, batch_counter)
        writer.add_scalar("Test/F1scr", F1scr_test, batch_counter)
        #mAP_raw_test, APs_raw_test = eval_map(pred_raw_all_test, y_all_test, avg="samples")
        #mAP_test, APs_test = eval_map(pred_all_test, y_all_test, avg="samples")
        apm_test.add(torch.from_numpy(pred_all_test), torch.from_numpy(y_all_test))
        mAP_test_cls = apm_val.value().cpu().detach().numpy()
        mAP_test = mAP_test_cls.mean()
        ltAnalysis(mAP_test_cls,batch_counter,"Test")
        writer.add_scalar("Test/mAP", mAP_test, batch_counter)

  if F1scr_val_new > F1scr_val:
    F1scr_val = F1scr_val_new 
    modelsavename = "weights_{}.weights".format(epoch_counter)
    torch.save(net.state_dict(), modelsavename)
    print("saved",modelsavename)
    patience = 0
  else:
    patience += 1


print("End")
