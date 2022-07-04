import math
import sys
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

from torch.utils import tensorboard
# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from lib.optim import build_optimizer, build_lr_scheduler
from lib.utils import code_backup, load_checkpoint, set_lr
import lib.models as models


class BaseTrainer(ABC):
    def __init__(self, config, resume, train_loader, save_dir, log_dir, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_validation = self.config.trainer.val
        self.start_epoch = 0

        # MODELS
        self.generator = getattr(models, config.generator.type)(**config.generator.args)
        self.discriminator = getattr(models, config.discriminator.type)(**config.discriminator.args)
        self.projection = getattr(models, config.projection.type)(**config.projection.args)

        # LOAD PRETRAINED NETWORK
        checkpoint_gen = self.config.trainer.get('pretrained_generator', False)
        if checkpoint_gen:
            checkpoint_gen = load_checkpoint(checkpoint_gen)
            self.generator.load_pretrained_weights(checkpoint_gen['state_dict'])

        checkpoint_disc = self.config.trainer.get('pretrained_discriminator', False)
        if checkpoint_disc:
            checkpoint_disc = load_checkpoint(checkpoint_disc)
            self.discriminator.load_pretrained_weights(checkpoint_disc['state_dict'])

        checkpoint_proj = self.config.trainer.get('pretrained_projection', False)
        if checkpoint_proj:
            checkpoint_proj = load_checkpoint(checkpoint_proj)
            self.projection.load_pretrained_weights(checkpoint_proj['state_dict'])

        # FREEZE LAYERS
        frozen_layers = self.config.generator.get('frozen_layers', None)
        if frozen_layers is not None:
            self.generator.set_trainable_specified_layers(frozen_layers, is_trainable=False)

        # SETTING THE DEVICE
        self.device, available_gpus = self._get_available_devices()
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.projection.to(self.device)

        # CONFIGS
        self.epochs = self.config.trainer.epochs
        self.save_period = self.config.trainer.save_period

        # OPTIMIZER
        self.gen_opt = build_optimizer(self.generator, optim=self.config.optimizer.type, **self.config.optimizer.args)
        self.gen_scheduler = build_lr_scheduler(self.gen_opt, lr_scheduler=self.config.lr_scheduler.type,
                                                max_epoch=self.epochs, **self.config.lr_scheduler.args)

        self.disc_opt = build_optimizer(self.discriminator, optim=self.config.optimizer.type, **self.config.optimizer.args)
        self.disc_scheduler = build_lr_scheduler(self.disc_opt, lr_scheduler=self.config.lr_scheduler.type,
                                                 max_epoch=self.epochs, **self.config.lr_scheduler.args)

        self.proj_opt = build_optimizer(self.projection, optim=self.config.optimizer.type, **self.config.optimizer.args)
        self.proj_scheduler = build_lr_scheduler(self.proj_opt, lr_scheduler=self.config.lr_scheduler.type,
                                                 max_epoch=self.epochs, **self.config.lr_scheduler.args)

        # CHECKPOINTS & TENSOBOARD
        self.writer = tensorboard.SummaryWriter(log_dir)
        self.checkpoint_dir = save_dir / 'checkpoints'
        self.visualize_dir = save_dir / 'images'
        self.code_dir = save_dir / 'code'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_dir.mkdir(parents=True, exist_ok=True)
        self.code_dir.mkdir(parents=True, exist_ok=True)
        code_backup(self.code_dir)

        # EARLY STOPPING
        self.min_delta = self.config.trainer.get('min_delta', 0.0)
        self.early_stop = self.config.trainer.get('early_stop', 50)
        self.monitor = self.config.trainer.get('monitor', "max")
        if self.monitor == 'min':
            self.is_better = lambda curr, best: curr < best - self.min_delta
        elif self.monitor == 'max':
            self.is_better = lambda curr, best: curr > best + self.min_delta
        elif self.monitor == 'none':
            self.is_better = lambda curr, best: False
            self.early_stop = float('inf')
        else:
            raise ValueError('Unexpected monitoring mode. It should be in ["min", "max"]')

        self.wrt_mode, self.wrt_step = 'train_', 0

        if resume:
            self._resume_checkpoint(resume)

        self.generator = torch.nn.DataParallel(self.generator, device_ids=available_gpus)
        self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=available_gpus)

    def train(self):
        best_metric = math.inf if self.monitor == 'min' else -math.inf
        num_bad_epochs = 0

        for epoch in range(self.start_epoch, self.epochs):
            # Train epoch
            self.wrt_mode = 'train'
            self._train_epoch(epoch)

            # Print Training Summary
            sys.stdout.description('\n' + self._training_summary(epoch)+'\n')

            if self.config.lr_scheduler.type != "reduce_plateau":
                self.gen_scheduler.step()
                self.disc_scheduler.step()
                self.proj_scheduler.step()

            # DO VALIDATION IF SPECIFIED
            if self.do_validation and (epoch+1) % self.config.trainer.val_per_epochs == 0 and self.val_loader is not None:
                sys.stdout.description('\n\n###### EVALUATION ######'+'\n')
                self.wrt_mode = 'val'
                metric = self._valid_epoch(epoch)

                # Print Validation Summary
                sys.stdout.description('\n' + self._validation_summary(epoch) + '\n')

                if self.config.lr_scheduler.type == "reduce_plateau":
                    self.gen_scheduler.step(metric)
                    self.disc_scheduler.step(metric)
                    self.proj_scheduler.step(metric)

                if self.is_better(metric, best_metric):
                    num_bad_epochs = 0
                    best_metric = metric
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    num_bad_epochs += 1

                if num_bad_epochs >= self.early_stop:
                    sys.stdout.description('\n EARLY STOPPED !!! \n')
                    break

            # SAVE CHECKPOINT
            if (epoch+1) % self.save_period == 0:
                self._save_checkpoint(epoch)

    def write_item(self, name, value, step):
        self.writer.add_scalar(f'{self.wrt_mode}/{name}', value, step)

    def _save_checkpoint(self, epoch, is_best=False, remove_module_from_keys=True, prefix='checkpoint'):
        state = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'generator_opt': self.gen_opt.state_dict(),
            'generator_scheduler': self.gen_scheduler.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_opt': self.disc_opt.state_dict(),
            'discriminator_scheduler': self.disc_scheduler.state_dict(),
            'projection_opt': self.proj_opt.state_dict(),
            'projection_schedular': self.proj_scheduler.state_dict(),
            'config': self.config
        }
        if remove_module_from_keys:
            def consume_prefix_in_state_dict_if_present(state_dict, prefix):
                keys = sorted(state_dict.keys())
                for key in keys:
                    if key.startswith(prefix):
                        newkey = key[len(prefix):]
                        state_dict[newkey] = state_dict.pop(key)

            consume_prefix_in_state_dict_if_present(state['generator_state_dict'], 'module.')
            consume_prefix_in_state_dict_if_present(state['discriminator_state_dict'], 'module.')

        if is_best:
            filename = self.checkpoint_dir / 'best_model.pth'
            torch.save(state, filename)
            sys.stdout.description("\nSaving current best: best_model.pth")
        else:
            filename = self.checkpoint_dir / f'{prefix}-epoch{epoch}.pth'
            torch.save(state, filename)
            sys.stdout.description(f'\nSaving a checkpoint: {filename} ...')

    def _resume_checkpoint(self, resume_path):
        print(f'Loading checkpoint : {resume_path}')
        checkpoint = load_checkpoint(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch']

        # MODELS
        if checkpoint['config']['generator']['type'] != self.config.generator.type:
            print({'Warning! Current generator is not the same as the one in the checkpoint'})
        self.generator.load_pretrained_weights(checkpoint['generator_state_dict'])

        if checkpoint['config']['discriminator']['type'] != self.config.discriminator.type:
            print({'Warning! Current discriminator is not the same as the one in the checkpoint'})
        self.discriminator.load_pretrained_weights(checkpoint['discriminator_state_dict'])

        # OPTIMIZERS
        self.gen_opt.load_state_dict(checkpoint['generator_opt'])
        self.disc_opt.load_state_dict(checkpoint['discriminator_opt'])
        self.proj_opt.load_state_dict(checkpoint['projection_opt'])

        self.gen_scheduler.load_state_dict(checkpoint['generator_scheduler'])
        self.disc_scheduler.load_state_dict(checkpoint['discriminator_scheduler'])
        self.proj_scheduler.load_state_dict(checkpoint['projection_scheduler'])

        print(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _get_available_devices(self):
        n_gpu = self.config.n_gpu
        gpu_id = self.config.get('gpu_id', 0)
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Number of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            torch.cuda.empty_cache()

        available_gpus = [gpu_id]
        device = torch.device(f'cuda:{gpu_id}' if n_gpu > 0 else 'cpu')
        print(f'Detected GPUs: {sys_gpu} Requested: {n_gpu} Selected GPU: {gpu_id}')

        return device, available_gpus

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _training_summary(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _validation_summary(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _log_train_tensorboard(self, step):
        raise NotImplementedError

    @abstractmethod
    def _log_validation_tensorboard(self, step):
        raise NotImplementedError

