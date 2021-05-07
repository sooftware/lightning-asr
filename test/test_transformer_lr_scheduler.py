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

import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim

from lightning_asr.optim.lr_scheduler import TransformerLRScheduler
from lightning_asr.optim.optimizer import Optimizer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.projection = nn.Linear(10, 10)

    def forward(self):
        pass


INIT_LR = 1e-10
PEAK_LR = 1e-04
FINAL_LR = 1e-07
INIT_LR_SCALE = 0.01
FINAL_LR_SCALE = 0.1
WARMUP_STEPS = 4000
MAX_GRAD_NORM = 400
TOTAL_STEPS = 120000

model = Model()

optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
scheduler = TransformerLRScheduler(
    optimizer=optimizer,
    peak_lr=PEAK_LR,
    final_lr=FINAL_LR,
    final_lr_scale=FINAL_LR_SCALE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=TOTAL_STEPS-WARMUP_STEPS,
)
optimizer = Optimizer(optimizer, scheduler, TOTAL_STEPS, MAX_GRAD_NORM)
lr_processes = list()

for timestep in range(TOTAL_STEPS):
    optimizer.step(model)
    lr_processes.append(optimizer.get_lr())

plt.title('Test Transformer lr scheduler')
plt.plot(lr_processes)
plt.xlabel('timestep', fontsize='large')
plt.ylabel('lr', fontsize='large')
plt.show()
