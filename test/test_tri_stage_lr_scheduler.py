import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim

from lasr.optim.lr_scheduler import TriStageLRScheduler
from lasr.optim.optimizer import Optimizer


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
FINAL_LR_SCALE = 0.001
WARMUP_STEPS = 4000
MAX_GRAD_NORM = 400
TOTAL_STEPS = 32000

model = Model()

optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
scheduler = TriStageLRScheduler(
    optimizer=optimizer,
    init_lr=INIT_LR,
    peak_lr=PEAK_LR,
    final_lr=FINAL_LR,
    init_lr_scale=INIT_LR_SCALE,
    final_lr_scale=FINAL_LR_SCALE,
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
)
optimizer = Optimizer(optimizer, scheduler, TOTAL_STEPS, MAX_GRAD_NORM)
lr_processes = list()

for timestep in range(TOTAL_STEPS):
    optimizer.step(model)
    lr_processes.append(optimizer.get_lr())

plt.title('Test Tri-stage lr scheduler')
plt.plot(lr_processes)
plt.xlabel('timestep', fontsize='large')
plt.ylabel('lr', fontsize='large')
plt.show()
