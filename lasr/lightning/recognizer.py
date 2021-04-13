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
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple, List

from lasr.model.decoder import DecoderRNN
from lasr.model.encoder import ConformerEncoder
from lasr.optim.lr_scheduler import TransformerLRScheduler
from lasr.criterioin.joint_ctc_cross_entropy import JointCTCCrossEntropyLoss
from lasr.optim.lr_scheduler.lr_scheduler import LearningRateScheduler


class LightningSpeechRecognizer(pl.LightningModule):
    """
    PyTorch Lightning ASR Pipeline. It consist of a conformer encoder and rnn decoder.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of model encoder
        num_encoder_layers (int, optional): Number of model blocks
        num_decoder_layers (int, optional): Number of decoder layers
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of model convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of model convolution module dropout
        decoder_dropout_p (float, optional): Probability of model decoder dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by model.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_decoder_layers: int = 2,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            decoder_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            max_length: int = 128,
            peak_lr: float = 1e-04,
            final_lr: float = 1e-07,
            final_lr_scale: float = 0.05,
            warmup_steps: int = 10000,
            decay_steps: int = 80000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            blank_id: int = 3,
            teacher_forcing_ratio: float = 1.0,
    ) -> None:
        super(LightningSpeechRecognizer, self).__init__()

        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.final_lr_scale = final_lr_scale
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.criterion = self.configure_criterion(num_classes, ignore_index=pad_id, blank_id=blank_id)

        self.encoder = ConformerEncoder(
            num_classes=num_classes,
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.decoder = DecoderRNN(
            num_classes=num_classes,
            max_length=max_length,
            hidden_state_dim=encoder_dim,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            num_heads=num_attention_heads,
            dropout_p=decoder_dropout_p,
            num_layers=num_decoder_layers,
            rnn_type="lstm",
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for inference.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        _, encoder_outputs, _ = self.encoder(inputs, input_lengths)
        predictions = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)
        return predictions

    def training_step(self, train_batch: tuple, batch_idx: int) -> Tensor:
        inputs, input_lengths, targets, target_lengths = train_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        predictions = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=self.teacher_forcing_ratio)

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=predictions.contiguous().view(-1, predictions.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        self.log("train_loss", loss)
        self.log("train_cross_entropy_loss", cross_entropy_loss)
        self.log("train_ctc_loss", ctc_loss)

        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int) -> Tensor:
        inputs, input_lengths, targets, target_lengths = val_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        predictions = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=0.0)

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=predictions.contiguous().view(-1, predictions.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        self.log("val_loss", loss)
        self.log("val_cross_entropy_loss", cross_entropy_loss)
        self.log("val_ctc_loss", ctc_loss)

        return loss

    def test_step(self, test_batch: tuple, batch_idx: int) -> Tensor:
        inputs, input_lengths, targets, target_lengths = test_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        predictions = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=0.0)

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=predictions.contiguous().view(-1, predictions.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        self.log("test_loss", loss)
        self.log("test_cross_entropy_loss", cross_entropy_loss)
        self.log("test_ctc_loss", ctc_loss)

        return loss

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[LearningRateScheduler]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = TransformerLRScheduler(
            optimizer,
            peak_lr=self.peak_lr,
            final_lr=self.final_lr,
            final_lr_scale=self.final_lr_scale,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
        )
        return [optimizer], [scheduler]

    def configure_criterion(self, num_classes: int, ignore_index: int, blank_id: int) -> nn.Module:
        criterion = JointCTCCrossEntropyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            reduction="mean",
            blank_id=blank_id,
            dim=-1,
            ctc_weight=0.3,
            cross_entropy_weight=0.7,
        )
        return criterion
