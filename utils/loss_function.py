"""
__author__      = 'kwok'
__time__        = '2021/7/30 10:34'
"""
from abc import ABC
import torch.nn as nn
import torch


class LabelSmoothingCrossEntropy(nn.Module, ABC):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        cls = pred.size(1)
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
