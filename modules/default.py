from pytorch_lightning import LightningModule
from utils import accuracy, rand_bbox
from models import get_model
from pytorch_lightning import LightningModule
import numpy as np
from torch import nn
import warmup_scheduler
import torch


class DefaultModule(LightningModule):
    def __init__(self, config, len_train=None):
        super().__init__()
        self.model = get_model(config)
        self.cutmix_beta = config.cutmix_beta
        self.cutmix_prob = config.cutmix_prob
        self.criterion = nn.NLLLoss()
        self.config = config
        self.save_hyperparameters()
        self.len_train = len_train

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx=None):
        images, target = batch

        output = self.model(images)
        loss = self.criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss)
        self.log("train_acc@1", acc1)
        self.log("train_acc@5", acc5)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch

        # compute output
        output = self(images)
        loss = self.criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.log("val_loss", loss)
        self.log("val_acc@1", acc1)
        self.log("val_acc@5", acc5)

        return loss


    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.config.lr)
                                         # weight_decay=self.config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epoch, eta_min=self.config.min_lr)
            # scheduler = torch.optim.lr_scheduler.LinearLR(
            #     optimizer=optimizer)
        elif self.config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.config.lr,
                                         weight_decay=self.config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epoch, eta_min=self.config.min_lr)
        elif self.config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config.lr,
                                        momentum=self.config.momentum,
                                        weight_decay=self.config.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.step, gamma=0.1)
        else:
            raise ValueError('Unknown optimizer')

        if self.config.warmup != 0:
            scheduler = warmup_scheduler.GradualWarmupScheduler(
                optimizer, multiplier=1., total_epoch=self.config.warmup, after_scheduler=scheduler)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }