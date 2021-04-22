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
from pytorch_lightning.loggers import TensorBoardLogger

from lasr.data.lit_data_module import LitLibriDataModule
from lasr.metric import WordErrorRate
from lasr.model.recognizer import LitSpeechRecognizer
from lasr.utils import check_environment


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def hydra_entry(configs: DictConfig) -> None:
    pl.seed_everything(configs.seed)

    logger = logging.getLogger(__name__)
    num_devices = check_environment(configs.use_cuda, logger)
    logger.info(OmegaConf.to_yaml(configs))

    if configs.use_tensorboard:
        logger = TensorBoardLogger("tensorboard", name="Lightning Automatic Speech Recognition")
    else:
        logger = True

    lit_data_module = LitLibriDataModule(configs)
    vocab = lit_data_module.prepare_data(configs.dataset_download, configs.vocab_size)
    lit_data_module.setup(vocab)

    model = LitSpeechRecognizer(
        configs=configs,
        num_classes=len(vocab),
        vocab=vocab,
        metric=WordErrorRate(vocab),
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
        logger=logger,
        auto_scale_batch_size=configs.auto_scale_batch_size,
    )
    trainer.fit(model, lit_data_module)


if __name__ == '__main__':
    hydra_entry()
