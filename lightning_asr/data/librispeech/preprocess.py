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
    'train-960',
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
]


def collect_transcripts(dataset_path, librispeech_dir: str = 'LibriSpeech'):
    """ Collect librispeech transcripts """
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
                                audio_path = os.path.join(librispeech_dir, dataset, subfolder1, subfolder2, tokens[0])
                                audio_path = f"{audio_path}.flac"
                                transcript = " ".join(tokens[1:])
                                dataset_transcripts.append('%s|%s' % (audio_path, transcript))

                    else:
                        continue

        transcripts_collection.append(dataset_transcripts)

    return transcripts_collection


def prepare_tokenizer(train_transcripts, vocab_size):
    """ Prepare sentencepice tokenizer """
    input_file = 'spm_input.txt'
    model_name = 'tokenizer'
    model_type = 'unigram'

    with open(input_file, 'w') as f:
        for transcript in train_transcripts:
            f.write('{}\n'.format(transcript.split('|')[-1]))

    cmd = f"--input={input_file} --model_prefix={model_name} --vocab_size={vocab_size} " \
          f"--model_type={model_type} --user_defined_symbols=<blank>"
    spm.SentencePieceTrainer.Train(cmd)


def generate_manifest_file(dataset_path: str, part: str, transcripts: list):
    """ Generate manifest file """
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer.model")

    with open(f"{dataset_path}/{part}.txt", 'w') as f:
        for transcript in transcripts:
            audio_path, transcript = transcript.split('|')
            text = " ".join(sp.EncodeAsPieces(transcript))
            label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])

            f.write('%s\t%s\t%s\n' % (audio_path, text, label))
