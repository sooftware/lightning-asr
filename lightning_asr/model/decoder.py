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

import random
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from torch import Tensor, LongTensor
from typing import Tuple, Optional

from lightning_asr.model.attention import MultiHeadAttention
from lightning_asr.model.modules import View


class DecoderRNN(nn.Module):
    """
    Converts higher level features (from encoder) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the decoder hidden state `h`
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        num_heads (int, optional): number of attention heads. (default: 4)
        num_layers (int, optional): number of recurrent layers (default: 2)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability of decoder (default: 0.2)
    """

    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            max_length: int = 128,
            hidden_state_dim: int = 1024,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3,
            use_tpu: bool = False,
    ) -> None:
        super(DecoderRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.use_tpu = use_tpu
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
            self,
            input_var: Tensor,
            hidden_states: Optional[Tensor],
            encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        if self.use_tpu:
            xla_device = xm.xla_device()
            targets = input_var.to(xla_device)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)
        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
            self,
            targets: Optional[Tensor] = None,
            encoder_outputs: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        hidden_states, attn = None, None
        predicted_log_probs = list()

        targets, batch_size, max_length = self._validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            step_outputs, hidden_states, attn = self.forward_step(targets, hidden_states, encoder_outputs)

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                predicted_log_probs.append(step_output)

        else:
            input_var = targets[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs)
                predicted_log_probs.append(step_outputs)
                input_var = predicted_log_probs[-1].topk(1)[1]

        predicted_log_probs = torch.stack(predicted_log_probs, dim=1)

        return predicted_log_probs

    def _validate_args(
            self,
            targets: Optional[Tensor] = None,
            encoder_outputs: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[Tensor, int, int]:
        """ Validate arguments """
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:  # inference
            targets = LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                targets = targets.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1  # minus the start of sequence symbol

        return targets, batch_size, max_length
