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
from omegaconf import DictConfig

from lasr.metric import WordErrorRate, ErrorRate
from lasr.model.decoder import DecoderRNN
from lasr.model.encoder import ConformerEncoder
from lasr.optim import AdamP, RAdam
from lasr.optim.lr_scheduler import TransformerLRScheduler, TriStageLRScheduler
from lasr.criterion.joint_ctc_cross_entropy import JointCTCCrossEntropyLoss
from lasr.optim.lr_scheduler.lr_scheduler import LearningRateScheduler
from lasr.optim.optimizer import Optimizer
from lasr.vocabs import Vocabulary, LibriSpeechVocabulary


class LightningSpeechRecognizer(pl.LightningModule):
    """
    PyTorch Lightning Speech Recognizer. It consist of a conformer encoder and rnn decoder.

    Args:
        configs (DictConfig): configuraion set
        num_classes (int): number of classification classes
        vocab (Vocabulary): vocab of training data
        metric (ErrorRate): performance measurement metrics

    Attributes:
        num_classes (int): Number of classification classes
        vocab (Vocabulary): vocab of training data
        peak_lr (float): peak learning rate
        final_lr (float): final learning rate
        init_lr_scale (float): scaling value of initial learning rate
        final_lr_scale (float): scaling value of final learning rate
        warmup_steps (int): warmup steps of learning rate
        decay_steps (int): decay steps of learning rate
        teacher_forcing_ratio (float): ratio of teacher forcing (forward label as decoder input)
        optimizer (str): name of optimizer (default: adam)
        lr_scheduler (str): name of learning rate scheduler (default: transformer)
    """
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = LibriSpeechVocabulary,
            metric: ErrorRate = WordErrorRate,
    ) -> None:
        super(LightningSpeechRecognizer, self).__init__()

        self.peak_lr = configs.peak_lr
        self.final_lr = configs.final_lr
        self.init_lr_scale = configs.init_lr_scale
        self.final_lr_scale = configs.final_lr_scale
        self.warmup_steps = configs.warmup_steps
        self.decay_steps = configs.decay_steps
        self.total_steps = configs.warmup_steps + configs.decay_steps
        self.max_grad_norm = configs.max_grad_norm
        self.teacher_forcing_ratio = configs.teacher_forcing_ratio
        self.vocab = vocab
        self.metric = metric
        self.optimizer = configs.optimizer
        self.lr_scheduler = configs.lr_scheduler
        self.criterion = self.configure_criterion(
            num_classes,
            ignore_index=self.vocab.pad_id,
            blank_id=self.vocab.blank_id,
            ctc_weight=configs.ctc_weight,
            cross_entropy_weight=configs.cross_entropy_weight,
        )

        self.encoder = ConformerEncoder(
            num_classes=num_classes,
            input_dim=configs.num_mels,
            encoder_dim=configs.encoder_dim,
            num_layers=configs.num_encoder_layers,
            num_attention_heads=configs.num_attention_heads,
            feed_forward_expansion_factor=configs.feed_forward_expansion_factor,
            conv_expansion_factor=configs.conv_expansion_factor,
            input_dropout_p=configs.input_dropout_p,
            feed_forward_dropout_p=configs.feed_forward_dropout_p,
            attention_dropout_p=configs.attention_dropout_p,
            conv_dropout_p=configs.conv_dropout_p,
            conv_kernel_size=configs.conv_kernel_size,
            half_step_residual=configs.half_step_residual,
            joint_ctc_attention=configs.joint_ctc_attention,
        )
        self.decoder = DecoderRNN(
            num_classes=configs.num_classes,
            max_length=configs.max_length,
            hidden_state_dim=configs.encoder_dim,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type,
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
        """ Configure optimizer """
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=self.lr)
        elif self.optimizer == 'radam':
            optimizer = RAdam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        if self.lr_scheduler == 'transformer':
            scheduler = TransformerLRScheduler(
                optimizer,
                peak_lr=self.peak_lr,
                final_lr=self.final_lr,
                final_lr_scale=self.final_lr_scale,
                warmup_steps=self.warmup_steps,
                decay_steps=self.decay_steps,
            )
        elif self.lr_scheduler == 'tri_stage':
            scheduler = TriStageLRScheduler(
                optimizer,
                init_lr=1e-10,
                peak_lr=self.peak_lr,
                final_lr=self.final_lr,
                final_lr_scale=self.final_lr_scale,
                init_lr_scale=self.init_lr_scale,
                warmup_steps=self.warmup_steps,
                total_steps=self.warmup_steps + self.decay_steps,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")

        optimizer = Optimizer(optimizer, scheduler, self.total_steps, self.max_grad_norm)

        return [optimizer], [scheduler]

    def configure_criterion(
            self,
            num_classes: int,
            ignore_index: int,
            blank_id: int,
            cross_entropy_weight: float,
            ctc_weight: float,
    ) -> nn.Module:
        """ Configure criterion """
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
