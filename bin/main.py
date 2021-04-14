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
from omegaconf import OmegaConf, DictConfig

from lasr.data.lit_data_module import LightningLibriDataModule
from lasr.metric import UnitErrorRate
from lasr.model.recognizer import LightningSpeechRecognizer
from lasr.vocabs import LibriSpeechVocabulary


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    pl.seed_everything(config.trainer.seed)

    vocab = LibriSpeechVocabulary(config.data.vocab_path, config.data.vocab_model_path)
    metric = UnitErrorRate(vocab)

    lit_data_module = LightningLibriDataModule(
        dataset_path=config.data.dataset_path,
        train_manifest_path=config.data.train_manifest_path,
        valid_clean_manifest_path=config.data.valid_clean_manifest_path,
        valid_other_manifest_path=config.data.valid_other_manifest_path,
        test_clean_manifest_path=config.data.test_clean_manifest_path,
        test_other_manifest_path=config.data.test_other_manifest_path,
        vocab=vocab,
        apply_spec_augment=config.data.spec_augment,
        num_epochs=config.data.num_epochs,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    lit_data_module.prepare_data()
    lit_data_module.setup()

    model = LightningSpeechRecognizer(
        num_classes=len(vocab),
        input_dim=config.recognizer.input_dim,
        encoder_dim=config.recognizer.encoder_dim,
        num_encoder_layers=config.recognizer.num_encoder_layers,
        num_decoder_layers=config.recognizer.num_decoder_layers,
        num_attention_heads=config.recognizer.num_attention_heads,
        feed_forward_expansion_factor=config.recognizer.feed_forward_expansion_factor,
        conv_expansion_factor=config.recognizer.conv_expansion_factor,
        input_dropout_p=config.recognizer.input_dropout_p,
        feed_forward_dropout_p=config.recognizer.feed_forward_dropout_p,
        conv_dropout_p=config.recognizer.conv_dropout_p,
        decoder_dropout_p=config.recognizer.decoder_dropout_p,
        conv_kernel_size=config.recognizer.conv_kernel_size,
        half_step_residual=config.recognizer.half_step_residual,
        max_length=config.recognizer.max_length,
        peak_lr=config.recognizer.peak_lr,
        final_lr=config.recognizer.final_lr,
        final_lr_scale=config.recognizer.final_lr_scale,
        warmup_steps=config.recognizer.warmup_steps,
        decay_steps=config.recognizer.decay_steps,
        vocab=vocab,
        metric=metric,
        teacher_forcing_ratio=config.recognizer.teacher_forcing_ratio,
        cross_entropy_weight=config.recognizer.cross_entropy_weight,
        ctc_weight=config.recognizer.ctc_weight,
        joint_ctc_attention=config.recognizer.joint_ctc_attention,
    )
    trainer = pl.Trainer(
        precision=config.trainer.precision,
        accelerator=config.trainer.accelerator,
        gpus=config.trainer.num_gpus,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        amp_backend=config.trainer.amp_backend,
        auto_select_gpus=config.trainer.auto_select_gpus,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        gradient_clip_val=config.trainer.gradient_clip_val,
    )
    trainer.fit(model, lit_data_module)


if __name__ == '__main__':
    main()
