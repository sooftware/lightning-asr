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


class AudioDataset(Dataset):
    """
    Dataset for audio & transcript matching

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        dataset_path (str): path of librispeech dataset
        audio_paths (list): list of audio path
        transcripts (list): list of transript
        apply_spec_augment (bool): flag indication whether to apply spec augment or not
        sos_id (int): identification of <sos>
        eos_id (int): identification of <eos>
        sample_rate (int): sampling rate of audio
        num_mels (int): the number of mfc coefficients to retain.
        frame_length (float): frame length for spectrogram (ms)
        frame_shift (float): length of hop between STFT (short time fourier transform) windows.
        freq_mask_para (int): hyper Parameter for freq masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
    """

    def __init__(
            self,
            dataset_path: str,
            audio_paths: list,
            transcripts: list,
            apply_spec_augment: bool = False,
            sos_id: int = 1,
            eos_id: int = 2,
            sample_rate: int = 16000,
            num_mels: int = 80,
            frame_length: float = 25.0,
            frame_shift: float = 10.0,
            freq_mask_para: int = 27,
            time_mask_num: int = 4,
            freq_mask_num: int = 2,
    ) -> None:
        super(AudioDataset, self).__init__()
        self.dataset_path = dataset_path
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.spec_augment_flags = [False] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))

        if apply_spec_augment:
            for idx in range(self.dataset_size):
                self.spec_augment_flags.append(True)
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

        self.shuffle()

    def _spec_augment(self, feature: Tensor) -> Tensor:
        """
        Provides Spec Augment. A simple data augmentation method for speech recognition.
        This concept proposed in https://arxiv.org/abs/1904.08779
        """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20      # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature

    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        """
        Provides feature extraction

        Inputs:
            signal (np.ndarray): audio signal

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        raise NotImplementedError

    def _parse_audio(self, audio_path: str, apply_spec_augment: bool) -> Tensor:
        """
        Parses audio.

        Args:
            audio_path (str): path of audio file
            apply_spec_augment (bool): flag indication whether to apply spec augment or not

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        signal = librosa.load(audio_path, sr=self.sample_rate)
        feature = self._get_feature(signal)

        feature -= feature.mean()
        feature /= np.std(feature)

        feature = FloatTensor(feature).transpose(0, 1)

        if apply_spec_augment:
            feature = self._spec_augment(feature)

        return feature

    def _parse_transcript(self, transcript: str) -> list:
        """
        Parses transcript

        Args:
            transcript (str): transcript of audio file

        Returns
            transcript (list): transcript that added <sos> and <eos> tokens
        """
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript

    def __getitem__(self, idx):
        """ Provides paif of audio & transcript """
        audio_path = os.path.join(self.dataset_path, self.audio_paths[idx])
        feature = self._parse_audio(audio_path, self.spec_augment_flags[idx])
        transcript = self._parse_transcript(self.transcripts[idx])
        return feature, transcript

    def shuffle(self):
        tmp = list(zip(self.audio_paths, self.transcripts, self.spec_augment_flags))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts, self.spec_augment_flags = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


class FBankDataset(AudioDataset):
    """ Dataset for filter bank & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        return torchaudio.compliance.kaldi.fbank(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.num_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        ).transpose(0, 1).numpy()


class SpectrogramDataset(AudioDataset):
    """ Dataset for spectrogram & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        spectrogram = torch.stft(
            Tensor(signal), self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=torch.hamming_window(self.n_fft),
            center=False, normalized=False, onesided=True
        )
        spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
        spectrogram = np.log1p(spectrogram.numpy())
        return spectrogram


class MelSpectrogramDataset(AudioDataset):
    """ Dataset for mel-spectrogram & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        melspectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=self.num_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram


class MFCCDataset(AudioDataset):
    """ Dataset for MFCC & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.num_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
