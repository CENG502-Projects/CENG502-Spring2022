import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb
from dataset import get_data_loader
from models.basic_layers import NeuralInterpreter
from classification_head import NeuralInterpreter_vision
from utils import *


############################# Main Hyperparameters ###############################
img_size =  32                          # Dimension of spatial axes of input images
patch_size = 4                          # Patch size
in_channels = 1                         # Dimension of input channels
embed_dim = 256                         # Dimension of embeddings
batch_size = 256                        # Number of batch
epochs = 100                            # Number of epochs
dim_c = 192                             # Dimension of 'code' vector
dim_inter = 192                         # Dimension of intermediate feature vector
ns = 2                                  # Number of 'scripts'
ni = 4                                  # Number of 'function' iterations
nl = 1                                  # Number of LOCs
nf = 5                                  # Number of 'function's
n_cls = 1                               # Number of CLS tokens
n_heads = 4                             # Number of heads per LOC
loc_features = 128                      # Number of features per LOC head
type_inference_depth = 2                # Type Inference MLP depth
type_inference_width = 192              # Type Inference MLP width 
treshold = 0.6                          # Trunctation Parameter
signature_dim = 24                      # Dimension of type_space
attn_prob = 0.0                         # Drop-out probability of ModAttn layer
proj_drop = 0.0                         # Drop-out probability of Projection 
mlp_depth = 4                           # Number of layers in ModMLP
number_of_class_mnist = 10              # Multi-class Classification class number 

############################## Optimizer Parameters ###############################          
beta1 = 0.9                             # Adam Optimizer beta1 parameter
beta2 = 0.999                           # Adam Optimizer beta2 parameter
lr = 1e-7                               # Learning Rate
warmup_steps = 20                       # Scheduler warm up steps

# Setting device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Parameters for dataset
datasetname = 'digits'
root = 'data/'

# Get dataloader
train_loader = get_data_loader(datasetname, root, batch_size)

# Create Neural Interpreter for vision Task
model = NeuralInterpreter_vision(ns, 
                                 ni, 
                                 nf, 
                                 embed_dim, 
                                 dim_c, 
                                 mlp_depth, 
                                 n_heads,
                                 type_inference_width, 
                                 signature_dim, 
                                 treshold, 
                                 dim_c, 
                                 n_classes=10,
                                 img_size=32, 
                                 patch_size=4, 
                                 in_channels=1, 
                                 n_cls=1,
                                 attn_prob=0, 
                                 proj_prob=0).to(device)

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
criterion = torch.nn.CrossEntropyLoss()
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs)

# log directory => save checkpoints
LOG_DIR = '/checkpoints_mnist/'

# Initialize wandb
wandb.init(project="Neural-Interpreter", entity="metugan")

# Run train
train(model, train_loader, valid_loader, criterion, optimizer, epochs, scheduler, LOG_DIR, device)
