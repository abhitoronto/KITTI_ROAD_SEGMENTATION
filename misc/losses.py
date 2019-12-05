"""
    This module contains loss definitions.
"""

import torch
import torch.nn as nn

from utils.utils import to_gpu

import numpy as np

class BCEJaccardLoss:
    """
        This loss consists from two parts: (1 - alpha) * BCE - alpha * log(Jaccard)
    """
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, targets, outputs):
        loss = (1 - self.alpha) * self.bce_loss(outputs, targets)

        if self.alpha:
            epsilon = 1e-15

            jaccard_targets = (targets == 1).float()
            jaccard_outputs = torch.sigmoid(outputs)

            intersection = (jaccard_targets * jaccard_outputs).sum()
            union = jaccard_targets.sum() + jaccard_outputs.sum()

            loss -= self.alpha * torch.log((intersection + epsilon) / (union - intersection + epsilon))
        return loss


class CCEJaccardLoss:
    """
        This is loss for multilable classification problem. Code was taken from https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py with a slightly modifications
    """
    def __init__(self, alpha=0, class_weights=None, num_classes=1):
        
        if class_weights is not None:
            nll_weight = to_gpu(torch.from_numpy(class_weights.astype(dtype=np.float32)))
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, targets, outputs):
        loss = (1 - self.alpha) * self.nll_loss(outputs, targets)

        if self.alpha:
            eps = 1e-15
            cls_weight = self.alpha / self.num_classes
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= cls_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss