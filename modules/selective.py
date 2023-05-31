from pytorch_lightning import LightningModule
from utils import accuracy, rand_bbox
from models import get_model
from .loss import SelectiveLoss
from .evaluator import Evaluator
from pytorch_lightning import LightningModule
import numpy as np
from torch import nn
import warmup_scheduler
import torch


class SelectiveModule(LightningModule):
    def __init__(self, config, len_train=None):
        super().__init__()
        self.model = get_model(config)
        self.cutmix_beta = config.cutmix_beta
        self.cutmix_prob = config.cutmix_prob

        base_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.selective_loss = SelectiveLoss(base_loss, coverage=config.coverage)
        self.ce_loss = nn.CrossEntropyLoss()
        self.config = config
        self.save_hyperparameters()
        self.len_train = len_train

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx=None):
        images, target = batch

        out_class, out_select, out_aux = self.model(images)

        selective_loss, loss_dict = self.selective_loss(out_class, out_select, target)
        selective_loss *= self.config.alpha

        ce_loss = self.ce_loss(out_aux, target)
        ce_loss *= (1.0 - self.config.alpha)
        
        # total loss
        loss = selective_loss + ce_loss

        evaluator = Evaluator(out_class.detach(), target.detach(), out_select.detach())
        result = evaluator()
        self.log("train_rejection_rate", result['rejection rate'])
        self.log("train_rejection_precision", result['rejection precision'])
        self.log("train_raw_acc", result['raw accuracy'])
        self.log("train_acc", result['accuracy'])

        acc1, acc5 = accuracy(out_class, target, topk=(1, 5))
        self.log("train_loss", loss)
        self.log("train_acc@1", acc1)
        self.log("train_acc@5", acc5)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch

        out_class, out_select, out_aux = self.model(images)
        selective_loss, loss_dict = self.selective_loss(out_class, out_select, target)
        selective_loss *= self.config.alpha

        ce_loss = self.ce_loss(out_aux, target)
        ce_loss *= (1.0 - self.config.alpha)
        
        # total loss
        loss = selective_loss + ce_loss

        evaluator = Evaluator(out_class.detach(), target.detach(), out_select.detach())
        result = evaluator()
        self.log("val_rejection_rate", result['rejection rate'])
        self.log("val_rejection_precision", result['rejection precision'])
        self.log("val_raw_acc", result['raw accuracy'])
        self.log("val_acc", result['accuracy'])

        acc1, acc5 = accuracy(out_class, target, topk=(1, 5))
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
                optimizer, step_size=self.config.step, gamma=self.config.gamma)
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
