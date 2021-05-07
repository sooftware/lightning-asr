# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Label smoothed cross entropy loss function.

    Args:
        num_classes (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
        smoothing (float): ratio of smoothing (confidence = 1.0 - smoothing)
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: sum)

    Inputs: logits, target
        logits (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label

    Returns: label_smoothed
        - **label_smoothed** (float): sum of loss
    """
    def __init__(
            self,
            num_classes: int,
            ignore_index: int,
            smoothing: float = 0.1,
            dim: int = -1,
            reduction='mean',
    ) -> None:
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()

        if self.reduction == 'sum':
            self.reduction_method = torch.sum
        elif self.reduction == 'mean':
            self.reduction_method = torch.mean
        else:
            raise ValueError("Unsupported reduction method {0}".format(reduction))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.smoothing > 0.0:
            with torch.no_grad():
                label_smoothed = torch.zeros_like(logits)
                label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
                label_smoothed.scatter_(1, targets.data.unsqueeze(1), self.confidence)
                label_smoothed[targets == self.ignore_index, :] = 0
            return self.reduction_method(-label_smoothed * logits)

        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction=self.reduction)