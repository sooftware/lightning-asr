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
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lightning_asr.vocabs.vocab import Vocabulary
from lightning_asr.vocabs import LibriSpeechVocabulary
from lightning_asr.data.dataset import (
    SpectrogramDataset,
    MelSpectrogramDataset,
    MFCCDataset,
    FBankDataset,
)
from lightning_asr.data.data_loader import (
    BucketingSampler,
    AudioDataLoader,
)
from lightning_asr.data.librispeech.preprocess import (
    collect_transcripts,
    prepare_tokenizer,
    generate_manifest_file,
)


def _parse_manifest_file(manifest_file_path: str) -> Tuple[list, list]:
    """ Parsing manifest file """
    audio_paths = list()
    transcripts = list()

    with open(manifest_file_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, _, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


class LightningLibriSpeechDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for LibriSpeech Dataset.

    Args:
        configs (DictConfig): configuraion set

    Attributes:
        dataset_path (str): path of librispeech dataset
        apply_spec_augment (bool): flag indication whether to apply spec augment or not
        num_epochs (int): the number of epochs
        batch_size (int): the size of batch samples
        num_workers (int): the number of cpu workers
        sample_rate (int): sampling rate of audio
        num_mels (int): the number of mfc coefficients to retain.
        frame_length (float): frame length for spectrogram (ms)
        frame_shift (float): length of hop between STFT (short time fourier transform) windows.
        freq_mask_para (int): hyper Parameter for freq masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
    """
    librispeech_parts = [
        'dev-clean',
        'test-clean',
        'dev-other',
        'test-other',
        'train-clean-100',
        'train-clean-360',
        'train-other-500',
    ]

    def __init__(self, configs: DictConfig) -> None:
        super(LightningLibriSpeechDataModule, self).__init__()
        self.dataset_path = configs.dataset_path
        self.librispeech_dir = 'LibriSpeech'
        self.manifest_paths = [
            f"{configs.dataset_path}/{self.librispeech_dir}/train-960.txt",
            f"{configs.dataset_path}/{self.librispeech_dir}/dev-clean.txt",
            f"{configs.dataset_path}/{self.librispeech_dir}/dev-other.txt",
            f"{configs.dataset_path}/{self.librispeech_dir}/test-clean.txt",
            f"{configs.dataset_path}/{self.librispeech_dir}/test-other.txt",
        ]
        self.dataset = dict()
        self.apply_spec_augment = configs.apply_spec_augment
        self.num_epochs = configs.num_epochs
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.sample_rate = configs.sample_rate
        self.num_mels = configs.num_mels
        self.frame_length = configs.frame_length
        self.frame_shift = configs.frame_shift
        self.freq_mask_para = configs.freq_mask_para
        self.time_mask_num = configs.time_mask_num
        self.freq_mask_num = configs.freq_mask_num
        self.logger = logging.getLogger(__name__)

        if configs.feature_extract_method == 'spectrogram':
            self.audio_dataset = SpectrogramDataset
        elif configs.feature_extract_method == 'melspectrogram':
            self.audio_dataset = MelSpectrogramDataset
        elif configs.feature_extract_method == 'mfcc':
            self.audio_dataset = MFCCDataset
        elif configs.feature_extract_method == 'fbank':
            self.audio_dataset = FBankDataset
        else:
            raise ValueError(f"Unsupported `feature_extract_method`: {configs.feature_extract_method}")

    def _download_librispeech(self) -> None:
        """
        Download librispeech dataset.
            - train-960(train-clean-100, train-clean-360, train-other-500)
            - dev-clean
            - dev-other
            - test-clean
            - test-other
        """
        base_url = "http://www.openslr.org/resources/12"
        train_dir = "train-960"

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        for part in self.librispeech_parts:
            self.logger.info(f"Librispeech-{part} download..")
            url = f"{base_url}/{part}.tar.gz"
            wget.download(url, self.dataset_path)

            self.logger.info(f"Un-tarring archive {self.dataset_path}/{part}.tar.gz")
            tar = tarfile.open(f"{self.dataset_path}/{part}.tar.gz", mode="r:gz")
            tar.extractall()
            tar.close()
            os.remove(f"{self.dataset_path}/{part}.tar.gz")

        self.logger.info("Merge all train packs into one")

        if not os.path.exists(os.path.join(self.dataset_path, self.librispeech_dir)):
            os.mkdir(os.path.join(self.dataset_path, self.librispeech_dir))
        if not os.path.exists(os.path.join(self.dataset_path, self.librispeech_dir, train_dir)):
            os.mkdir(os.path.join(self.dataset_path, self.librispeech_dir, train_dir))

        for part in self.librispeech_parts[:-3]:    # dev, test
            shutil.move(
                os.path.join(self.librispeech_dir, part),
                os.path.join(self.dataset_path, self.librispeech_dir, part),
            )

        for part in self.librispeech_parts[-3:]:    # train
            path = os.path.join(self.librispeech_dir, part)
            subfolders = os.listdir(path)
            for subfolder in subfolders:
                shutil.move(
                    os.path.join(path, subfolder),
                    os.path.join(self.dataset_path, self.librispeech_dir, train_dir, subfolder),
                )

    def _generate_manifest_files(self, vocab_size: int) -> None:
        """
        Generate manifest files.
        Format: {audio_path}\t{transcript}\t{numerical_label}

        Args:
            vocab_size (int): size of subword vocab

        Returns:
            None
        """
        self.logger.info("Generate Manifest Files..")
        transcripts_collection = collect_transcripts(self.dataset_path)
        prepare_tokenizer(transcripts_collection[0], vocab_size)

        for idx, part in enumerate(['train-960', 'dev-clean', 'dev-other', 'test-clean', 'test-other']):
            generate_manifest_file(self.dataset_path, part, transcripts_collection[idx])

    def prepare_data(self, download: bool = False, vocab_size: int = 5000) -> Vocabulary:
        """
        Prepare librispeech data

        Args:
            download (bool): if True, download librispeech dataset
            vocab_size (int): size of subword vocab

        Returns:
            None
        """
        if download:
            self._download_librispeech()
        self._generate_manifest_files(vocab_size)
        return LibriSpeechVocabulary("tokenizer.model", vocab_size)

    def setup(self, vocab: Vocabulary) -> None:
        """ Split dataset into train, valid, and test. """
        splits = ['train', 'val-clean', 'val-other', 'test-clean', 'test-other']

        for idx, (path, split) in enumerate(zip(self.manifest_paths, splits)):
            audio_paths, transcripts = _parse_manifest_file(path)
            self.dataset[split] = self.audio_dataset(
                dataset_path=self.dataset_path,
                audio_paths=audio_paths,
                transcripts=transcripts,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                apply_spec_augment=self.apply_spec_augment if idx == 0 else False,
                sample_rate=self.sample_rate,
                num_mels=self.num_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                freq_mask_para=self.freq_mask_para,
                freq_mask_num=self.freq_mask_num,
                time_mask_num=self.time_mask_num,
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
