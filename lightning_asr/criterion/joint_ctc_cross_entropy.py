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

import torch.nn as nn
from typing import Tuple
from torch import Tensor


class JointCTCCrossEntropyLoss(nn.Module):
    """
    Privides Joint CTC-CrossEntropy Loss function

    Args:
        num_classes (int): the number of classification
        ignore_index (int): indexes that are ignored when calculating loss
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: mean)
        ctc_weight (float): weight of ctc loss
        cross_entropy_weight (float): weight of cross entropy loss
        blank_id (int): identification of blank for ctc
    """
    def __init__(
            self,
            num_classes: int,
            ignore_index: int,
            dim: int = -1,
            reduction='mean',
            ctc_weight: float = 0.3,
            cross_entropy_weight: float = 0.7,
            blank_id: int = None,
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction, zero_infinity=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(
            self,
            encoder_log_probs: Tensor,
            decoder_log_probs: Tensor,
            output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctc_loss = self.ctc_loss(encoder_log_probs, targets, output_lengths, target_lengths)
        print(decoder_log_probs.size())
        print(targets.size())
        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1))
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss, ctc_loss, cross_entropy_loss
