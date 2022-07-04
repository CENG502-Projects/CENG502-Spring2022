from datetime import datetime
import torch 

train_dataset_path = "./dataset_org/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/"
lt_dataset_output_path = "./dataset_voc_lt/"

test_dataset_path = "./dataset_org/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/"
test_dataset_output_path = "./dataset_voc_test/"
test_weight_path = "src/weight/weights_22000.weights"

debug_mode = False

seed = 123 # datetime.now()

# Initials
if torch.cuda.is_available():  
  devtype = "cuda:0" 
else:  
  devtype = "cpu"  
dev = torch.device(devtype) 

iteration = 100000
test_batch_size = 8
batch_size_ = 8
lr_ = 0.01
validation_set_ratio = 0.1
momentum_ = 0.9
weight_decay_ = 0.0001
lambda_ = 0.1
save_weight_interval = 100
patience_level = 8
threshold = torch.tensor([0.5]).to(dev)
tnsrbrd_dir = "./runs"

uniform_branch_active = True 
resampled_branch_active = True 
logit_consistency = True 
logit_compensation = True # TODO
aug_test = True # TODO


voc_labels = {
    "aeroplane":0,
    "bicycle":1,
    "bird":2,
    "boat":3,
    "bottle":4,
    "bus":5,
    "car":6,
    "cat":7,
    "chair":8,
    "cow":9,
    "diningtable":10,
    "dog":11,
    "horse":12,
    "motorbike":13,
    "person":14,
    "pottedplant":15,
    "sheep":16,
    "sofa":17,
    "train":18,
    "tvmonitor":19
}

labels_map = {
    0:"aeroplane",
    1:"bicycle",
    2:"bird",
    3:"boat",
    4:"bottle",
    5:"bus",
    6:"car",
    7:"cat",
    8:"chair",
    9:"cow",
    10:"diningtable",
    11:"dog",
    12:"horse",
    13:"motorbike",
    14:"person",
    15:"pottedplant",
    16:"sheep",
    17:"sofa",
    18:"train",
    19:"tvmonitor"
}

lt = {
  "head":["person","dog","car","chair","cat","bird"],
  "medium":["bottle","aeroplane","tvmonitor","train","diningtable","pottedplant"],
  "tail":["bicycle","motorbike","boat","sofa","horse","bus","sheep","cow"]
}