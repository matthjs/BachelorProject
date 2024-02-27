from torchrl.data import ReplayBuffer

import torch.nn as nn
from trainers.rltrainer import RLTrainer


class DQNTrainer(RLTrainer):
    def __init__(self, model: nn.Module, batch_size: int, buf: ReplayBuffer, learning_rate: float = 0.01,
                 discount_factor=0.9):
        super().__init__(model, batch_size, buf, learning_rate, discount_factor)