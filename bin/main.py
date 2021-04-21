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

import os
import hydra
import pytorch_lightning as pl
import logging
from omegaconf import OmegaConf, DictConfig

from lasr.data.lit_data_module import LightningLibriDataModule
from lasr.metric import WordErrorRate
from lasr.model.recognizer import LightningSpeechRecognizer
from lasr.utils import check_environment
from lasr.vocabs import LibriSpeechVocabulary


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def main(configs: DictConfig) -> None:
    pl.seed_everything(configs.seed)
    logger = logging.getLogger(__name__)
    num_devices = check_environment(configs.use_cuda, logger)
    logger.info(OmegaConf.to_yaml(configs))

    lit_data_module = LightningLibriDataModule(
        dataset_path=configs.dataset_path,
        feature_extract_method=configs.feature_extract_method,
        apply_spec_augment=configs.spec_augment,
        num_epochs=configs.num_epochs,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        sample_rate=configs.sample_rate,
        num_mels=configs.num_mels,
        frame_length=configs.frame_length,
        frame_shift=configs.frame_shift,
        freq_mask_para=configs.freq_mask_para,
        time_mask_num=configs.time_mask_num,
        freq_mask_num=configs.freq_mask_num,
    )
    lit_data_module.prepare_data(configs.dataset_download, configs.vocab_size)
    vocab = LibriSpeechVocabulary("tokenizer.model", configs.vocab_size)
    metric = WordErrorRate(vocab)
    lit_data_module.setup(vocab)

    model = LightningSpeechRecognizer(
        num_classes=len(vocab),
        input_dim=configs.input_dim,
        encoder_dim=configs.encoder_dim,
        num_encoder_layers=configs.num_encoder_layers,
        num_decoder_layers=configs.num_decoder_layers,
        num_attention_heads=configs.num_attention_heads,
        feed_forward_expansion_factor=configs.feed_forward_expansion_factor,
        conv_expansion_factor=configs.conv_expansion_factor,
        input_dropout_p=configs.input_dropout_p,
        feed_forward_dropout_p=configs.feed_forward_dropout_p,
        conv_dropout_p=configs.conv_dropout_p,
        decoder_dropout_p=configs.decoder_dropout_p,
        conv_kernel_size=configs.conv_kernel_size,
        half_step_residual=configs.half_step_residual,
        max_length=configs.max_length,
        peak_lr=configs.peak_lr,
        final_lr=configs.final_lr,
        init_lr_scale=configs.init_lr_scale,
        final_lr_scale=configs.final_lr_scale,
        warmup_steps=configs.warmup_steps,
        decay_steps=configs.decay_steps,
        max_grad_norm=configs.gradient_clip_val,
        vocab=vocab,
        metric=metric,
        teacher_forcing_ratio=configs.teacher_forcing_ratio,
        cross_entropy_weight=configs.cross_entropy_weight,
        ctc_weight=configs.ctc_weight,
        joint_ctc_attention=configs.joint_ctc_attention,
        optimizer=configs.optimizer,
        lr_scheduler=configs.lr_scheduler,
    )
    trainer = pl.Trainer(
        precision=configs.precision,
        accelerator=configs.accelerator,
        gpus=num_devices,
        accumulate_grad_batches=configs.accumulate_grad_batches,
        amp_backend=configs.amp_backend,
        auto_select_gpus=configs.auto_select_gpus,
        check_val_every_n_epoch=configs.check_val_every_n_epoch,
        gradient_clip_val=configs.gradient_clip_val,
    )
    trainer.fit(model, lit_data_module)


if __name__ == '__main__':
    main()
