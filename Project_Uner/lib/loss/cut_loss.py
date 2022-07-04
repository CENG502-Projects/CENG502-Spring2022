import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.patchnce import PatchNCELoss


class CUTLoss(nn.Module):
    def __init__(self, generator, discriminator, projector, config, device):
        super().__init__()

        self.lambdaGAN = config.trainer.get('gan_lambda', 1.0)
        self.lambdaNCE = config.trainer.get('nce_lambda', 1.0)
        self.nce_layers = config.trainer.get('nce_layers', [0, 4, 8, 12, 16])
        self.num_patches = config.loss.get('num_patches', 256)
        self.batch_size = config.datamanager.batch_size_train

        self.generator = generator
        self.discriminator = discriminator
        self.projector = projector

        self.criterionNCE = []
        for _ in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(self.batch_size).to(device))

    def adv_loss(self, pred, is_real):
        target = torch.ones_like if is_real else torch.zeros_like
        return F.mse_loss(pred, target(pred))

    def forward(self, real_A, real_B):
        fake_B = self.generator(real_A)
        idt_B = self.generator(real_B)

        # discriminator
        fake_preds_for_d = self.discriminator(fake_B.detach())
        real_preds_for_d = self.discriminator(real_B)

        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) +
            self.adv_loss(fake_preds_for_d, False)
        )

        # generator
        fake_preds_for_g = self.discriminator(fake_B)
        g_gan_loss = self.adv_loss(fake_preds_for_g, True) * self.lambdaGAN
        g_nce_loss = self.nce_loss(real_A, fake_B)
        g_nce_loss += self.nce_loss(real_B, idt_B)
        g_loss = g_gan_loss + (0.5 * g_nce_loss)

        return g_loss, d_loss, fake_B.detach(), idt_B.detach()

    def nce_features(self, x):
        feat = x
        feats = []
        for layer_id, layer in enumerate(self.generator.module.model):
            feat = layer(feat)
            if layer_id in self.nce_layers:
                feats.append(feat)
            if len(feats) == len(self.nce_layers):
                break
        return feats

    def nce_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.nce_features(tgt)
        feat_k = self.nce_features(src)
        feat_k_pool, sample_ids = self.projector(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.projector(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.lambdaNCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers