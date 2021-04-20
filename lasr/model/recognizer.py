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

from lasr.metric import WordErrorRate, ErrorRate
from lasr.model.decoder import DecoderRNN
from lasr.model.encoder import ConformerEncoder
from lasr.optim.lr_scheduler import TransformerLRScheduler
from lasr.criterioin.joint_ctc_cross_entropy import JointCTCCrossEntropyLoss
from lasr.optim.lr_scheduler.lr_scheduler import LearningRateScheduler
from lasr.vocabs import Vocabulary, LibriSpeechVocabulary


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
        max_length (int, optional): max decoding length
        peak_lr (float): peak learning rate
        final_lr (float): final learning rate
        final_lr_scale (float): scaling value of final learning rate
        warmup_steps (int): warmup steps of learning rate
        decay_steps (int): decay steps of learning rate
        vocab (Vocabulary): vocab of training data
        teacher_forcing_ratio (float): ratio of teacher forcing (forward label as decoder input)
        cross_entropy_weight (float): weight of cross entropy loss
        ctc_weight (float): weight of ctc loss
        joint_ctc_attention (bool): flag indication joint ctc attention or not
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
            vocab: Vocabulary = LibriSpeechVocabulary,
            metric: ErrorRate = WordErrorRate,
            teacher_forcing_ratio: float = 1.0,
            cross_entropy_weight: float = 0.7,
            ctc_weight: float = 0.3,
            joint_ctc_attention: bool = True,
    ) -> None:
        super(LightningSpeechRecognizer, self).__init__()

        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.final_lr_scale = final_lr_scale
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.vocab = vocab
        self.metric = metric
        self.criterion = self.configure_criterion(
            num_classes,
            ignore_index=self.vocab.pad_id,
            blank_id=self.vocab.blank_id,
            ctc_weight=ctc_weight,
            cross_entropy_weight=cross_entropy_weight,
        )

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
            joint_ctc_attention=joint_ctc_attention,
        )
        self.decoder = DecoderRNN(
            num_classes=num_classes,
            max_length=max_length,
            hidden_state_dim=encoder_dim,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            num_heads=num_attention_heads,
            dropout_p=decoder_dropout_p,
            num_layers=num_decoder_layers,
            rnn_type="lstm",
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for inference.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * y_hats (torch.FloatTensor): Result of model predictions.
        """
        _, encoder_outputs, _ = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)
        return y_hats

    def training_step(self, train_batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `input_lengths`, `targets`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, input_lengths, targets, target_lengths = train_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=self.teacher_forcing_ratio)

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        uer = self.metric(targets, y_hats)

        self.log("train_uer", uer)
        self.log("train_loss", loss)
        self.log("train_cross_entropy_loss", cross_entropy_loss)
        self.log("train_ctc_loss", ctc_loss)

        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `input_lengths`, `targets`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, input_lengths, targets, target_lengths = val_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=0.0)

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        uer = self.metric(targets, y_hats)

        self.log("val_uer", uer)
        self.log("val_loss", loss)
        self.log("val_cross_entropy_loss", cross_entropy_loss)
        self.log("val_ctc_loss", ctc_loss)

        return loss

    def test_step(self, test_batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `input_lengths`, `targets`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, input_lengths, targets, target_lengths = test_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=0.0)

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        uer = self.metric(targets, y_hats)

        self.log("test_uer", uer)
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

    def configure_criterion(
            self,
            num_classes: int,
            ignore_index: int,
            blank_id: int,
            cross_entropy_weight: float,
            ctc_weight: float,
    ) -> nn.Module:
        criterion = JointCTCCrossEntropyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            reduction="mean",
            blank_id=blank_id,
            dim=-1,
            cross_entropy_weight=cross_entropy_weight,
            ctc_weight=ctc_weight,
        )
        return criterion
