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
from omegaconf import DictConfig

from lightning_asr.data.librispeech.lit_data_module import LightningLibriSpeechDataModule
from lightning_asr.metric import WordErrorRate, CharacterErrorRate
from lightning_asr.model import LightningASRModel
from lightning_asr.utilities import parse_configs


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def hydra_entry(configs: DictConfig) -> None:
    pl.seed_everything(configs.seed)
    logger, num_devices = parse_configs(configs)

    data_module = LightningLibriSpeechDataModule(configs)
    vocab = data_module.prepare_data(configs.dataset_download, configs.vocab_size)
    data_module.setup(vocab=vocab)

    model = LightningASRModel(
        configs=configs,
        num_classes=len(vocab),
        vocab=vocab,
        wer=WordErrorRate(vocab),
        cer=CharacterErrorRate(vocab),
    )

    if configs.use_tpu:
        trainer = pl.Trainer(
            tpu_cores=configs.tpu_cores,
            accelerator=configs.accelerator,
            accumulate_grad_batches=configs.accumulate_grad_batches,
            check_val_every_n_epoch=configs.check_val_every_n_epoch,
            gradient_clip_val=configs.gradient_clip_val,
            logger=logger,
            auto_scale_batch_size=configs.auto_scale_batch_size,
            max_epochs=configs.max_epochs,
        )
    else:
        trainer = pl.Trainer(
            precision=configs.precision,
            accelerator=configs.accelerator,
            gpus=num_devices,
            accumulate_grad_batches=configs.accumulate_grad_batches,
            amp_backend=configs.amp_backend,
            auto_select_gpus=configs.auto_select_gpus,
            check_val_every_n_epoch=configs.check_val_every_n_epoch,
            gradient_clip_val=configs.gradient_clip_val,
            logger=logger,
            auto_scale_batch_size=configs.auto_scale_batch_size,
            max_epochs=configs.max_epochs,
        )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    hydra_entry()
