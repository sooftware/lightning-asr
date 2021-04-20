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

import argparse

from lasr.data.preprocess import (
    collect_transcripts,
    prepare_tokenizer,
    generate_transcript_file
)

LIBRI_SPEECH_DATASETS = [
    'train_960',
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
]


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='LibriSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='your_dataset_path',
                        help='path of original dataset')
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        help='size of vocab')

    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    transcripts_collection = collect_transcripts(opt.dataset_path)
    prepare_tokenizer(transcripts_collection[0], opt.vocab_size)

    for idx, dataset in enumerate(LIBRI_SPEECH_DATASETS):
        generate_transcript_file(dataset, transcripts_collection[idx])


if __name__ == '__main__':
    main()
