from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer

from trainers.trainer import AbstractTrainer


# NOT USED!


class RLTrainer(AbstractTrainer, ABC):
    """
    Base class for RL optimizers.
    """

    def __init__(self, models: dict, batch_size: int, buf: ReplayBuffer, learning_rate: float, discount_factor=0.9,
                 loss=nn.SmoothL1Loss):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.models = models  # make sure model is on GPU!
        self.loss_fn = loss

    def set_replay_buffer(self, buf: ReplayBuffer):
        self.buf = buf
