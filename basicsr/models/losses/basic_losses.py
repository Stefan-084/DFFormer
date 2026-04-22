import torch
from torch import nn as nn
from torch.nn import functional as F

from .loss_util import weighted_loss

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


class FreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight=1.0, reduction=reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.0
            for predi in pred:
                diff = torch.fft.rfft2(predi) - torch.fft.rfft2(target)
                loss = loss + torch.mean(torch.abs(diff)) * 0.01 + self.l1_loss(predi, target)
            return self.loss_weight * loss / len(pred)

        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        freq_term = torch.mean(torch.abs(diff)) * 0.01
        return self.loss_weight * (freq_term + self.l1_loss(pred, target))
