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
import sentencepiece as spm

LIBRI_SPEECH_DATASETS = [
    'train_960',
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
]


def collect_transcripts(dataset_path):
    transcripts_collection = list()

    for dataset in LIBRI_SPEECH_DATASETS:
        dataset_transcripts = list()

        for subfolder1 in os.listdir(os.path.join(dataset_path, dataset)):
            for subfolder2 in os.listdir(os.path.join(dataset_path, dataset, subfolder1)):
                for file in os.listdir(os.path.join(dataset_path, dataset, subfolder1, subfolder2)):
                    if file.endswith('txt'):
                        with open(os.path.join(dataset_path, dataset, subfolder1, subfolder2, file)) as f:
                            for line in f.readlines():
                                tokens = line.split()
                                audio = '%s.flac' % os.path.join(dataset, subfolder1, subfolder2, tokens[0])
                                transcript = " ".join(tokens[1:])
                                dataset_transcripts.append('%s|%s' % (audio, transcript))

                    else:
                        continue

        transcripts_collection.append(dataset_transcripts)

    return transcripts_collection


def prepare_tokenizer(train_transcripts, vocab_size):
    input_file = 'spm_input.txt'
    model_name = 'tokenizer'
    model_type = 'unigram'

    with open(input_file, 'w') as f:
        for transcript in train_transcripts:
            f.write('{}\n'.format(transcript.split('|')[-1]))

    input_args = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s'
    cmd = input_args % (input_file, model_name, vocab_size, model_type)
    spm.SentencePieceTrainer.Train(cmd)


def generate_transcript_file(dataset_name, transcripts):
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer.model")

    with open('../../data/%s-transcript.txt' % dataset_name, 'w') as f:
        for transcript in transcripts:
            audio, transcript = transcript.split('|')
            text = " ".join(sp.EncodeAsPieces(transcript))
            label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])

            f.write('%s\t%s\t%s\n' % (audio, text, label))


def merge_train_dev_transcript_file():
    merge_list = ['train_960', 'dev-clean', 'dev-other']

    lines = list()

    for dataset in merge_list:
        with open('../../data/%s-transcript.txt' % dataset) as f:
            for line in f.readlines():
                lines.append(line)

    with open('../../data/train.txt', 'w') as f:
        for line in lines:
            f.write('%s' % line)
