from abc import ABC, abstractmethod

from torchrl.data import ReplayBuffer

from trainers.trainer import Trainer


class RLTrainer(Trainer, ABC):
    """
    Base class for RL optimizers.
    """
    @abstractmethod
    def set_replay_buffer(self, buf: ReplayBuffer):
        pass
