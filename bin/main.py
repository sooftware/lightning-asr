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
from lasr.vocabs import LibriSpeechVocabulary


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(config))

    lit_data_module = LightningLibriDataModule(
        dataset_path=config.dataset_path,
        feature_extract_method=config.feature_extract_method,
        apply_spec_augment=config.spec_augment,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sample_rate=config.sample_rate,
        num_mels=config.num_mels,
        frame_length=config.frame_length,
        frame_shift=config.frame_shift,
        freq_mask_para=config.freq_mask_para,
        time_mask_num=config.time_mask_num,
        freq_mask_num=config.freq_mask_num,
    )
    lit_data_module.prepare_data(config.dataset_download, config.vocab_size)
    vocab = LibriSpeechVocabulary("tokenizer.model", config.vocab_size)
    metric = WordErrorRate(vocab)
    lit_data_module.setup(vocab)

    model = LightningSpeechRecognizer(
        num_classes=len(vocab),
        input_dim=config.input_dim,
        encoder_dim=config.encoder_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        feed_forward_expansion_factor=config.feed_forward_expansion_factor,
        conv_expansion_factor=config.conv_expansion_factor,
        input_dropout_p=config.input_dropout_p,
        feed_forward_dropout_p=config.feed_forward_dropout_p,
        conv_dropout_p=config.conv_dropout_p,
        decoder_dropout_p=config.decoder_dropout_p,
        conv_kernel_size=config.conv_kernel_size,
        half_step_residual=config.half_step_residual,
        max_length=config.max_length,
        peak_lr=config.peak_lr,
        final_lr=config.final_lr,
        init_lr_scale=config.init_lr_scale,
        final_lr_scale=config.final_lr_scale,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        vocab=vocab,
        metric=metric,
        teacher_forcing_ratio=config.teacher_forcing_ratio,
        cross_entropy_weight=config.cross_entropy_weight,
        ctc_weight=config.ctc_weight,
        joint_ctc_attention=config.joint_ctc_attention,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )
    trainer = pl.Trainer(
        precision=config.precision,
        accelerator=config.accelerator,
        gpus=config.num_gpus,
        accumulate_grad_batches=config.accumulate_grad_batches,
        amp_backend=config.amp_backend,
        auto_select_gpus=config.auto_select_gpus,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
    )
    trainer.fit(model, lit_data_module)


if __name__ == '__main__':
    main()
