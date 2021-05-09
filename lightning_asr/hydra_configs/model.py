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

from dataclasses import dataclass


@dataclass
class ConformerLSTMConfigs:
    encoder_dim: int = 256
    num_encoder_layers: int = 17
    num_decoder_layers: int = 2
    num_attention_heads: int = 8
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    decoder_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    half_step_residual: bool = True
    max_length: int = 128
    teacher_forcing_ratio: float = 1.0
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    joint_ctc_attention: bool = True
    rnn_type: str = "lstm"
    optimizer: str = "adam"
