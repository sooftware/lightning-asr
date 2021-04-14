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

# $1 : DIR_TO_SAVE_DATA

base_url=www.openslr.org/resources/12
train_dir=train_960
vocab_size=5000

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <download_dir>>"
  exit 1
fi

download_dir=${1%/}

echo "Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    url=$base_url/$part.tar.gz
    if ! wget -P $download_dir $url; then
        echo "$0: wget failed for $url"
        exit 1
    fi
    if ! tar -C $download_dir -xvzf $download_dir/$part.tar.gz; then
        echo "$0: error un-tarring archive $download_dir/$part.tar.gz"
        exit 1
    fi
done

echo "Merge all train packs into one"
mkdir -p ${download_dir}/LibriSpeech/${train_dir}/
for part in train-clean-100 train-clean-360 train-other-500; do
    mv ${download_dir}/LibriSpeech/${part}/* $download_dir/LibriSpeech/${train_dir}/
done

python prepare-libri.py --dataset_path $1/LibriSpeech --vocab_size $vocab_size

for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    rm $part.tar.gz
done
