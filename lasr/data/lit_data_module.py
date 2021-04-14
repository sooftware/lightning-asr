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
import pytorch_lightning as pl
from typing import Any, Union, List, Tuple
from torch.utils.data import DataLoader

from lasr.data.data_loader import SpectrogramDataset, BucketingSampler, AudioDataLoader
from lasr.vocabs import Vocabulary


def _parse_manifest_file(manifest_file_path: str) -> Tuple[list, list]:
    audio_paths = list()
    transcripts = list()

    with open(manifest_file_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, _, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


class LightningLibriDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_path: str,
            train_manifest_path: str,
            valid_clean_manifest_path: str,
            valid_other_manifest_path: str,
            test_clean_manifest_path: str,
            test_other_manifest_path: str,
            vocab: Vocabulary,
            apply_spec_augment: bool,
            num_epochs: int,
            batch_size: int,
            num_workers: int,
    ) -> None:
        super(LightningLibriDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.manifest_paths = [
            train_manifest_path,
            valid_clean_manifest_path,
            valid_other_manifest_path,
            test_clean_manifest_path,
            test_other_manifest_path,
        ]
        self.dataset = dict()
        self.vocab = vocab
        self.apply_spec_augment = apply_spec_augment
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        for path in self.manifest_paths:
            if not os.path.isfile(path):
                raise FileExistsError("Manifest file is not exist.")

    def setup(self):
        splits = ['train', 'val-clean', 'val-other', 'test-clean', 'test-other']

        for idx, (path, split) in enumerate(zip(self.manifest_paths, splits)):
            audio_paths, transcripts = _parse_manifest_file(path)
            self.dataset[split] = SpectrogramDataset(
                dataset_path=self.dataset_path,
                audio_paths=audio_paths,
                transcripts=transcripts,
                sos_id=self.vocab.sos_id,
                eos_id=self.vocab.eos_id,
                apply_spec_augment=self.apply_spec_augment if idx == 0 else False,
            )

    def train_dataloader(self) -> Any:
        train_sampler = BucketingSampler(self.dataset['train'], batch_size=self.batch_size)
        return AudioDataLoader(self.dataset['train'], num_workers=self.num_workers, batch_sampler=train_sampler)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        valid_clean_sampler = BucketingSampler(self.dataset['val-clean'], batch_size=self.batch_size)
        valid_other_sampler = BucketingSampler(self.dataset['val-other'], batch_size=self.batch_size)
        return [
            AudioDataLoader(self.dataset['val-clean'], num_workers=self.num_workers, batch_sampler=valid_clean_sampler),
            AudioDataLoader(self.dataset['val-other'], num_workers=self.num_workers, batch_sampler=valid_other_sampler),
        ]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_clean_sampler = BucketingSampler(self.dataset['test-clean'], batch_size=self.batch_size)
        test_other_sampler = BucketingSampler(self.dataset['test-other'], batch_size=self.batch_size)
        return [
            AudioDataLoader(self.dataset['test-clean'], num_workers=self.num_workers, batch_sampler=test_clean_sampler),
            AudioDataLoader(self.dataset['test-other'], num_workers=self.num_workers, batch_sampler=test_other_sampler),
        ]
