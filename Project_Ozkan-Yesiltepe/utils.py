import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import math
from torch.optim.lr_scheduler import LambdaLR

import wandb
from tqdm import tqdm
import os 
# Cosine Learning Rate Scheduler adapted from: https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
	
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss

best_acc = 0

def train(net, train_loader, valid_loader, criterion, optimizer, epochs, scheduler, LOG_DIR='.', device='cpu'):
    wandb.watch(net, criterion, log="all", log_freq=10)

    terminate = EarlyStopping(patience=10, verbose=False, delta=0.1)

    # Run training and track with wandb
    for epoch in tqdm(range(epochs)):
        print('\nEpoch: %d' % epoch)
        ##################################= TRAING =##################################
        net.train()
        train_loss_class = 0
        train_correct = 0
        train_total = 0
        
        step_idx = 0
        for batch_idx, (inputs, class_targets) in enumerate(train_loader):
            inputs, class_targets = inputs.to(device), class_targets.to(device)
            optimizer.zero_grad()
            
            outputs_class = net(inputs)

            class_loss = criterion(outputs_class, class_targets)

            class_loss.backward()
            optimizer.step()

            train_loss_class += class_loss.item()

            _, predicted = outputs_class.max(1)
            train_total += class_targets.size(0)
            train_correct += predicted.eq(class_targets).sum().item()

            step_idx +=1
            if step_idx % 25 == 0:
                train_loss_class = train_loss_class/(step_idx)
                train_acc = 100.*train_correct/train_total
                
                visualize_loss = {
                'train_loss': train_loss_class,
                'train_acc': train_acc,
                  }
                wandb.log(visualize_loss, step=batch_idx)
                
                step_idx = 0
                train_loss_class = 0
                train_correct = 0
                train_total = 0
                
        ##################################= VALIDATION =##################################
        global best_acc
        net.eval()
        valid_loss_class = 0

        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch_idx, (inputs, class_targets) in enumerate(valid_loader):
                inputs, class_targets = inputs.to(device), class_targets.to(device)
                
                outputs_class = net(inputs)

                class_loss = criterion(outputs_class, class_targets)

                valid_loss_class += class_loss.item()

                _, predicted = outputs_class.max(1)
                valid_total += class_targets.size(0)
                valid_correct += predicted.eq(class_targets).sum().item()

            # progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        valid_loss_class = valid_loss_class/(batch_idx+1)
        
        valid_acc = 100.*valid_correct/valid_total

        ##################################= WANDB-LOG + CHECKPOINT =##################################
        # Save checkpoint.
        if valid_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': valid_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(LOG_DIR+'checkpoint'):
                os.makedirs(LOG_DIR + 'checkpoint')
            torch.save(state, LOG_DIR + 'checkpoint/ckpt.pth')
            best_acc = valid_acc
        
        visualize_loss = {
                'valid_loss': valid_loss_class,
                'valid_acc': valid_acc,
                  }
        wandb.log(visualize_loss, step=epoch)

        # wandb-log
        # loss_class = [train_loss_class, valid_loss_class]
        # acc_class = [train_acc, valid_acc]

        # loss_log(train_loss_class, valid_loss_class, train_acc, valid_acc, epoch)
        scheduler.step()

        # Early-Stop Check
        terminate(valid_loss_class, net)
        if terminate.early_stop:
            print("Early stopping...")
            break

def loss_log(train_loss_class, valid_loss_class, train_acc, valid_acc, epoch):
    # Where the magic happens
    wandb.log({ \
        "epoch": epoch+1, "train_acc": train_acc, "valid_acc": valid_acc \
        , "train_loss_class": train_loss_class, "valid_loss_class": valid_loss_class})
    
    print(f"Loss after " + f" Epochs: {epoch+1:.3f}" + f" train_loss_class: {train_loss_class:.3f}" + f" valid_loss_class: {valid_loss_class:.3f}" \
    + f" train_acc: {train_acc:.3f}" + f" valid_acc: {valid_acc:.3f}")

      
      
def visualize_data(loader):
  '''
  Visualize the data in a grid.
  '''
  # Get a sample of data
  samples = next(iter(loader))[0]
  
  # Determine grid size
  grid_width = 8 
  grid_height = 8
  
  # Create the grid
  image_grid = make_grid(samples[:grid_width*grid_height], nrow=grid_width)
  
  # Visualization arrangement
  plt.rcParams['figure.figsize'] = [grid_height, grid_width]
  plt.imshow(image_grid.permute(1, 2, 0))
  plt.axis('off')
  plt.show()  
  
 
