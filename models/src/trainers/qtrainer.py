from abc import ABC

from torchrl.data import ReplayBuffer

from trainers.rltrainer import RLTrainer
from trainers.trainer import Trainer


class QTrainer(RLTrainer):
    def __init__(self, model, batch_size: int, buf: ReplayBuffer, learning_rate: float, discount_factor=0.9):
        super().__init__(model, batch_size, buf, learning_rate, discount_factor)

    def train(self) -> None:
        batch = self.buf.sample(batch_size=self.batch_size)
        print(batch)
