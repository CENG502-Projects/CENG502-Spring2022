import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from lib.base import BaseTrainer
from lib.utils import AverageMeter
from lib.loss.cut_loss import CUTLoss


class CUTTrainer(BaseTrainer):
    def __init__(self, config, resume, train_loader, save_dir, log_dir, val_loader=None):
        super().__init__(config, resume, train_loader, save_dir, log_dir, val_loader)

        self.train_vis = self.config.trainer.get('visualize_train_batch', False)
        self.val_vis = self.config.trainer.get('visualize_val_batch', False)
        self.vis_count = self.config.trainer.get('vis_count', len(self.train_loader))
        self.log_per_batch = self.config.trainer.get('log_per_batch', int(np.sqrt(self.train_loader.batch_size)))
        self.loss_fn = CUTLoss(self.generator, self.discriminator, self.projection, config, device=self.device)

    def _train_epoch(self, epoch):
        vis_save_dir = self.visualize_dir / 'train' / str(epoch)
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        self.generator.train()
        self.discriminator.train()
        self.projection.train()
        train_vis_count = 0
        tic = time.time()
        self._reset_metrics()

        tbar = tqdm(self.train_loader)
        for batch_idx, data in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            batches_done = (epoch-1) * len(self.train_loader) + batch_idx
            real_A = data['A'].to(self.device)
            real_B = data['B'].to(self.device)

            g_loss, d_loss, fake_B, idt_B = self.loss_fn(real_A, real_B)

            self.gen_opt.zero_grad()
            self.proj_opt.zero_grad()
            g_loss.backward()
            self.gen_opt.step()
            self.proj_opt.step()

            self.disc_opt.zero_grad()
            d_loss.backward()
            self.disc_opt.step()

            # update metrics
            self.loss_meter.update(g_loss.item() + d_loss.item())
            self.gen_loss_meter.update(g_loss.item())
            self.disc_loss_meter.update(d_loss.item())
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # Visualize batch & Tensorboard log
            if batch_idx % self.log_per_batch == 0:
                self._log_train_tensorboard(batches_done)
                if train_vis_count < self.vis_count and self.train_vis:
                    train_vis_count += real_B.shape[0]
                    self._visualize_batch(real_A, real_B, fake_B, idt_B, batch_idx, vis_save_dir)
            tbar.set_description(self._training_summary(epoch))

    def _valid_epoch(self, epoch):
        vis_save_dir = self.visualize_dir / 'test' / str(epoch)
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        self.generator.eval()
        self.discriminator.eval()
        self.projection.eval()
        self._reset_metrics()
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            val_vis_count = 0
            for batch_idx, data in enumerate(tbar):
                real_A = data['A'].to(self.device)
                real_B = data['B'].to(self.device)

                # train discriminator
                fake_B = self.generator(real_A)
                idt_B = self.generator(real_B)

                # Visualize batch
                if val_vis_count < self.vis_count and self.val_vis:
                    val_vis_count += real_A.shape[0]
                    self._visualize_batch(real_A, real_B, fake_B, idt_B, batch_idx, vis_save_dir)

                # PRINT INFO
                if batch_idx == len(tbar)-1:
                    tbar.set_description(self._validation_summary(epoch))

            self._log_validation_tensorboard(epoch)
        return self.loss_meter.avg

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter = AverageMeter()
        self.disc_loss_meter = AverageMeter()
        self.gen_loss_meter = AverageMeter()
        self.l1_loss_meter = AverageMeter()
        self.vgg_loss_meter = AverageMeter()

    def _log_train_tensorboard(self, step):
        self.write_item(name='gen_loss', value=self.gen_loss_meter.avg, step=step)
        self.write_item(name='disc_loss', value=self.disc_loss_meter.avg, step=step)

        for i, opt_group in enumerate(self.gen_opt.param_groups):
            self.write_item(name=f'Learning_rate_generator_{i}', value=opt_group['lr'], step=self.wrt_step)

        for i, opt_group in enumerate(self.disc_opt.param_groups):
            self.write_item(name=f'Learning_rate_discriminator{i}', value=opt_group['lr'], step=self.wrt_step)

    def _log_validation_tensorboard(self, step):
        self.write_item(name='loss', value=self.loss_meter.avg, step=step)

    def _training_summary(self, epoch):
        return f'TRAIN [{epoch}] ' \
               f'Loss: {self.loss_meter.val:.3f}({self.loss_meter.avg:.3f}) | ' \
               f'DISC: {self.disc_loss_meter.val:.3f}({self.disc_loss_meter.avg:.3f}) | ' \
               f'GEN: {self.gen_loss_meter.val:.3f}({self.gen_loss_meter.avg:.3f}) | ' \
               f'gen_lr {self.gen_opt.param_groups[0]["lr"]:.6f} | ' \
               f'disc_lr {self.disc_opt.param_groups[0]["lr"]:.6f} | ' \
               f'b {self.batch_time.avg:.2f} D {self.data_time.avg:.2f}'

    def _validation_summary(self, epoch):
        return f'EVAL [{epoch}] | '

    def _visualize_batch(self, real_A, real_B, fake_B, idt_B, step, vis_save_dir, vis_shape=(128, 128), predef=''):
        ra = self.train_loader.dataset.denormalize(real_A.clone(), device=self.device)
        ra = F.interpolate(ra, size=vis_shape, mode='bilinear', align_corners=True)

        rb = self.train_loader.dataset.denormalize(real_B.clone(), device=self.device)
        rb = F.interpolate(rb, size=vis_shape, mode='bilinear', align_corners=True)

        fb = self.train_loader.dataset.denormalize(fake_B.clone(), device=self.device)
        fb = F.interpolate(fb, size=vis_shape, mode='bilinear', align_corners=True)

        ib = self.train_loader.dataset.denormalize(idt_B.clone(), device=self.device)
        ib = F.interpolate(ib, size=vis_shape, mode='bilinear', align_corners=True)

        vis_img = torch.cat((ra, rb, fb, ib), dim=-1)
        save_image(vis_img, str(vis_save_dir / f'{predef}_index_{step}.png'), nrow=1)