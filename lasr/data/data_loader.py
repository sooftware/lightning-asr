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
import random
import librosa
import torch
import torchaudio
import numpy as np
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class SpectrogramDataset(Dataset):
    """ Dataset for feature & transcript matching """
    feature_configs = {
        'sample_rate': 16000,
        'n_mels': 80,
        'frame_length': 25.0,
        'frame_shift': 10.0,
    }
    spec_augment_configs = {
        'freq_mask_para': 27,
        'time_mask_num': 4,
        'freq_mask_num': 2,
    }

    def __init__(
            self,
            dataset_path: str,
            audio_paths: list,
            transcripts: list,
            apply_spec_augment: bool = False,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(SpectrogramDataset, self).__init__()
        self.dataset_path = dataset_path
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.spec_augment_flag = [False] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id

        if apply_spec_augment:
            for idx in range(self.dataset_size):
                self.spec_augment_flag.append(True)
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

        self.shuffle()

    def _spec_augment(self, feature: Tensor) -> Tensor:
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20      # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.spec_augment_configs['time_mask_num']):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.spec_augment_configs['freq_mask_num']):
            f = int(np.random.uniform(low=0.0, high=self.spec_augment_configs['freq_mask_para']))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature

    def _parse_audio(self, audio_path: str, apply_spec_augment: bool) -> Tensor:
        signal = librosa.load(audio_path, sr=self.feature_configs['sample_rate'])
        fbank = torchaudio.compliance.kaldi.fbank(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.feature_configs['n_mels'],
            frame_length=self.feature_configs['frame_length'],
            frame_shift=self.feature_configs['frame_shift'],
        ).transpose(0, 1).numpy()

        fbank -= fbank.mean()
        fbank /= np.std(fbank)

        fbank = FloatTensor(fbank).transpose(0, 1)

        if apply_spec_augment:
            fbank = self._spec_augment(fbank)

        return fbank

    def _parse_transcript(self, transcript):
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript

    def __getitem__(self, idx):
        feature = self._parse_audio(os.path.join(self.dataset_path, self.audio_paths[idx]), self.spec_augment_flag[idx])
        transcript = self._parse_transcript(self.transcripts[idx])
        return feature, transcript

    def shuffle(self):
        tmp = list(zip(self.audio_paths, self.transcripts, self.spec_augment_flag))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts, self.spec_augment_flag = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    seq_lengths = [s[0].size(1) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_size = max(seq_lengths)
    max_target_size = max(target_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size: int = 32) -> None:
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
