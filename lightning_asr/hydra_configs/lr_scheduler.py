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
class LRSchedulerConfigs:
    lr: float = 1e-04


@dataclass
class ReduceLROnPlateauLRSchedulerConfigs(LRSchedulerConfigs):
    lr_scheduler: str = "reduce_lr_on_plateau"
    lr_patience: int = 1
    lr_factor: float = 0.3


@dataclass
class TriStageLRSchedulerConfigs(LRSchedulerConfigs):
    lr_scheduler: str = "tri_stage"
    init_lr: float = 1e-10
    peak_lr: float = 1e-04
    final_lr: float = 1e-07
    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.05
    warmup_steps: int = 10000
    decay_steps: int = 150000


@dataclass
class TransformerLRSchedulerConfigs(LRSchedulerConfigs):
    lr_scheduler: str = "transformer"
    peak_lr: float = 1e-04
    final_lr: float = 1e-07
    final_lr_scale: float = 0.05
    warmup_steps: int = 10000
    decay_steps: int = 150000
