from torch.optim.lr_scheduler import _LRScheduler

class StepLRWithWarmup(_LRScheduler):
  def __init__(self, optimizer, warmup_epochs, base_lr, warmup_lr, decay_epochs, gamma, last_epoch=-1):
    self.warmup_epochs = warmup_epochs
    self.base_lr = base_lr
    self.warmup_lr = warmup_lr
    self.decay_epochs = decay_epochs
    self.gamma = gamma
    super(StepLRWithWarmup, self).__init__(optimizer, last_epoch)

  def get_step_lr(self):
    if len(self.decay_epochs) != 0:
      if self.last_epoch == self.decay_epochs[0]:
        self.decay_epochs.remove(self.decay_epochs[0])
        return [self.gamma * lr for lr in self.get_last_lr()]
    return self.get_last_lr()

  def get_warmup_lr(self):
    diff = self.warmup_lr - self.base_lr
    warmup_step = diff / self.warmup_epochs
    return [self.base_lr + warmup_step * self.last_epoch for _ in self.base_lrs]

  def get_lr(self):
    if self.last_epoch <= self.warmup_epochs:
        return self.get_warmup_lr()
    else:
        return self.get_step_lr()
