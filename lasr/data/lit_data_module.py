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
import wget
import tarfile
import logging
import shutil
import pytorch_lightning as pl
from typing import Union, List, Tuple
from torch.utils.data import DataLoader

from lasr.vocabs import Vocabulary
from lasr.data.data_loader import (
    SpectrogramDataset,
    BucketingSampler,
    AudioDataLoader,
)
from lasr.data.libri_preprocess import (
    collect_transcripts,
    prepare_tokenizer,
    generate_transcript_file,
)


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
            apply_spec_augment: bool,
            num_epochs: int,
            batch_size: int,
            num_workers: int,
    ) -> None:
        super(LightningLibriDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.manifest_paths = [
            f"{dataset_path}/train-960-transcript.txt",
            f"{dataset_path}/dev-clean-transcript.txt",
            f"{dataset_path}/dev-other-transcript.txt",
            f"{dataset_path}/test-clean-transcript.txt",
            f"{dataset_path}/test-other-transcript.txt",
        ]
        self.dataset = dict()
        self.apply_spec_augment = apply_spec_augment
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, download: bool = False, vocab_size: int = 5000) -> None:
        base_url = "http://www.openslr.org/resources/12"
        train_dir = "train_960"
        dataset_parts = [
            'dev-clean', 'test-clean', 'dev-other', 'test-other',
            'train-clean-100', 'train-clean-360', 'train-other-500',
        ]

        if download:
            if not os.path.exists(self.dataset_path):
                os.mkdir(self.dataset_path)

            for part in dataset_parts:
                self.logger.info(f"librispeech-{part} download..")
                url = f"{base_url}/{part}.tar.gz"
                wget.download(url, self.dataset_path)

                self.logger.info(f"un-tarring archive {self.dataset_path}/{part}.tar.gz")
                tar = tarfile.open(f"{self.dataset_path}/{part}.tar.gz", mode="r:gz")
                tar.extractall()
                tar.close()

            self.logger.info("Merge all train packs into one")

            if not os.path.exists(f"{self.dataset_path}/LibriSpeech/"):
                os.mkdir(f"{self.dataset_path}/LibriSpeech/")
            if not os.path.exists(f"{self.dataset_path}/LibriSpeech/{train_dir}/"):
                os.mkdir(f"{self.dataset_path}/LibriSpeech/{train_dir}/")

            for part in dataset_parts[-3:]:
                path = f"{self.dataset_path}/LibriSpeech/{part}/"
                files = os.listdir(path)
                for file in files:
                    shutil.move(f"{path}/{file}", f"{self.dataset_path}/LibriSpeech/{train_dir}/{file}")

        self.logger.info("Data Pre-processing..")
        transcripts_collection = collect_transcripts(f"{self.dataset_path}/LibriSpeech/")
        prepare_tokenizer(transcripts_collection[0], vocab_size)

        for idx, part in enumerate(['train_960', 'dev-clean', 'dev-other', 'test-clean', 'test-other']):
            generate_transcript_file(self.dataset_path, part, transcripts_collection[idx])

        self.logger.info("Remove .tar.gz files..")
        for part in dataset_parts:
            os.remove(f"{self.dataset_path}/{part}.tar.gz")

    def setup(self, vocab: Vocabulary) -> None:
        splits = ['train', 'val-clean', 'val-other', 'test-clean', 'test-other']

        for idx, (path, split) in enumerate(zip(self.manifest_paths, splits)):
            audio_paths, transcripts = _parse_manifest_file(path)
            self.dataset[split] = SpectrogramDataset(
                dataset_path=self.dataset_path,
                audio_paths=audio_paths,
                transcripts=transcripts,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                apply_spec_augment=self.apply_spec_augment if idx == 0 else False,
            )

    def train_dataloader(self) -> DataLoader:
        train_sampler = BucketingSampler(self.dataset['train'], batch_size=self.batch_size)
        return AudioDataLoader(
            dataset=self.dataset['train'],
            num_workers=self.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_clean_sampler = BucketingSampler(self.dataset['val-clean'], batch_size=self.batch_size)
        val_other_sampler = BucketingSampler(self.dataset['val-other'], batch_size=self.batch_size)
        return [
            AudioDataLoader(
                dataset=self.dataset['val-clean'],
                num_workers=self.num_workers,
                batch_sampler=val_clean_sampler,
            ),
            AudioDataLoader(
                dataset=self.dataset['val-other'],
                num_workers=self.num_workers,
                batch_sampler=val_other_sampler,
            ),
        ]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_clean_sampler = BucketingSampler(self.dataset['test-clean'], batch_size=self.batch_size)
        test_other_sampler = BucketingSampler(self.dataset['test-other'], batch_size=self.batch_size)
        return [
            AudioDataLoader(
                dataset=self.dataset['test-clean'],
                num_workers=self.num_workers,
                batch_sampler=test_clean_sampler,
            ),
            AudioDataLoader(
                dataset=self.dataset['test-other'],
                num_workers=self.num_workers,
                batch_sampler=test_other_sampler,
            ),
        ]
