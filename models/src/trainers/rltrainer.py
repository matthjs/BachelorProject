from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer

from trainers.trainer import Trainer


class RLTrainer(Trainer, ABC):
    """
    Base class for RL optimizers.
    """

    def __init__(self, model, batch_size: int, buf: ReplayBuffer, learning_rate: float, discount_factor=0.9,
                 loss=nn.MSELoss()):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.model = model  # make sure model is on GPU!
        self.loss_fn = loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

    def set_replay_buffer(self, buf: ReplayBuffer):
        self.buf = buf
