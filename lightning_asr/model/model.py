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
from typing import Dict, Union
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, Adadelta, Adagrad, SGD, Adamax, AdamW, ASGD

from lightning_asr.metric import WordErrorRate, ErrorRate, CharacterErrorRate
from lightning_asr.model.decoder import DecoderRNN
from lightning_asr.model.encoder import ConformerEncoder
from lightning_asr.optim import AdamP, RAdam
from lightning_asr.optim.lr_scheduler import TransformerLRScheduler, TriStageLRScheduler
from lightning_asr.criterion import JointCTCCrossEntropyLoss
from lightning_asr.vocabs import LibriSpeechVocabulary
from lightning_asr.vocabs.vocab import Vocabulary


class LightningASRModel(pl.LightningModule):
    """
    PyTorch Lightning Automatic Speech Recognition Model. It consist of a conformer encoder and rnn decoder.

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
            wer: ErrorRate = WordErrorRate,
            cer: ErrorRate = CharacterErrorRate,
    ) -> None:
        super(LightningASRModel, self).__init__()

        self.peak_lr = configs.peak_lr
        self.final_lr = configs.final_lr
        self.init_lr_scale = configs.init_lr_scale
        self.final_lr_scale = configs.final_lr_scale
        self.warmup_steps = configs.warmup_steps
        self.decay_steps = configs.decay_steps
        self.total_steps = configs.warmup_steps + configs.decay_steps
        self.gradient_clip_val = configs.gradient_clip_val
        self.teacher_forcing_ratio = configs.teacher_forcing_ratio
        self.vocab = vocab
        self.wer = wer
        self.cer = cer
        self.optimizer = configs.optimizer
        self.lr = configs.lr
        self.lr_scheduler = configs.lr_scheduler
        self.lr_patience = configs.lr_patience
        self.lr_factor = configs.lr_factor
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
            num_classes=num_classes,
            max_length=configs.max_length,
            hidden_state_dim=configs.encoder_dim,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type,
            use_tpu=configs.use_tpu,
        )

    def _log_states(
            self,
            stage: str,
            wer: float,
            cer: float,
            loss: float,
            cross_entropy_loss: float,
            ctc_loss: float,
    ) -> None:
        self.log(f"{stage}_wer", wer)
        self.log(f"{stage}_cer", cer)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        self.log(f"{stage}_ctc_loss", ctc_loss)
        return

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
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = train_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=self.teacher_forcing_ratio)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = y_hats[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        wer = self.wer(targets[:, 1:], y_hats)
        cer = self.cer(targets[:, 1:], y_hats)

        self._log_states('train', wer, cer, loss, cross_entropy_loss, ctc_loss)

        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int, dataset_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = val_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = y_hats[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        wer = self.wer(targets[:, 1:], y_hats)
        cer = self.cer(targets[:, 1:], y_hats)

        self._log_states('valid', wer, cer, loss, cross_entropy_loss, ctc_loss)

        return loss

    def test_step(self, test_batch: tuple, batch_idx: int, dataset_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = test_batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = y_hats[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        wer = self.wer(targets[:, 1:], y_hats)
        cer = self.cer(targets[:, 1:], y_hats)

        self._log_states('test', wer, cer, loss, cross_entropy_loss, ctc_loss)

        return loss

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, object, str]]:
        """ Configure optimizer """
        supported_optimizers = {
            "adam": Adam,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
        }
        assert self.optimizer in supported_optimizers.keys(), f"Unsupported Optimizer: {self.optimizer}"
        optimizer = supported_optimizers[self.optimizer](self.parameters(), lr=self.lr)

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
        elif self.lr_scheduler == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
            )
        else:
            raise ValueError(f"Unsupported `lr_scheduler`: {self.lr_scheduler}")

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'metric_to_track',
        }

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
