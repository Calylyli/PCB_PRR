
import numpy as np
from mindspore import Tensor
from src.model_utils.config import config


def get_lr(lr_init, lr_max, total_steps, warmup_steps, decay_steps):
    """get the learning rate of each step"""
    decay_step_index = list(range(decay_steps, total_steps, decay_steps))
    decay_step_index.append(total_steps) # pivot for convenience
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            for j in range(len(decay_step_index)):
                if i < decay_step_index[j]:
                    lr = lr_max * pow(config.decay_rate, j)
                    break
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = Tensor(lr_each_step)
    return lr_each_step
