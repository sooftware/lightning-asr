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
class BaseTrainerConfigs:
    seed: int = 1
    accelerator: str = "dp"
    precision: int = 16
    accumulate_grad_batches: int = 4
    amp_backend: str = "apex"
    num_workers: int = 4
    batch_size: int = 32
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 5.0
    use_tensorboard: bool = True
    max_epochs: int = 20
    auto_scale_batch_size: str = "binsearch"


@dataclass
class TrainerGPUConfigs(BaseTrainerConfigs):
    use_cuda: bool = True
    use_tpu: bool = False
    auto_select_gpus: bool = True


@dataclass
class TrainerTPUConfigs(BaseTrainerConfigs):
    use_cuda: bool = False
    use_tpu: bool = True
    tpu_cores: int = 8
