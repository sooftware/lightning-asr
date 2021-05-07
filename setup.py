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

from distutils.core import setup

setup(
    name='lightning_asr',
    version='latest',
    description='Modular and extensible speech recognition library leveraging pytorch-lightning and hydra',
    author='Soohwan Kim',
    author_email='kaki.ai@tunib.ai',
    url='https://github.com/sooftware/lightning_asr',
    install_requires=[
        'torch>=1.4.0',
        'python-Levenshtein',
        'librosa >= 0.7.0',
        'torchaudio',
        'numpy',
        'pandas',
        'astropy',
        'sentencepiece',
        'pytorch-lightning',
        'hydra-core',
        'wget',
    ],
    keywords=['asr', 'speech_recognition', 'pytorch-lightning'],
    python_requires='>=3.7',
)
