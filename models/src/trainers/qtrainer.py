from abc import ABC

from trainers.rltrainer import RLTrainer
from trainers.trainer import Trainer


class QTrainer(RLTrainer):

    def train(self) -> None:
        pass
