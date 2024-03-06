from abc import ABC, abstractmethod

import torchrl
from torch import nn

from builders.abstractagentbuilder import BaseAgentBuilder
from exploration.egreedy import EpsilonGreedy
from trainers.rltrainer import RLTrainer


class ValueAgentBuilder(BaseAgentBuilder, ABC):
    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.learning_rate = None
        self.discount_factor = None
        self.replay_buffer_size = None
        self.annealing_num_steps = None

        self.value_model_type_str = None
        self.buffer_storage_type = None

        self.exploration_policy = None
        self.trainer = None

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        return self

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
        return self

    def set_discount_factor(self, discount_factor: float):
        self.discount_factor = discount_factor
        return self

    def set_replay_buffer_size(self, replay_buffer_size: int):
        self.replay_buffer_size = replay_buffer_size
        return self

    def set_annealing_num_steps(self, annealing_num_steps: int):
        self.annealing_num_steps = annealing_num_steps
        return self

    def set_buffer_storage_type(self, buffer_storage_type: type[torchrl.data.replay_buffers.Storage]):
        self.buffer_storage_type = buffer_storage_type
        return self

    def set_value_model_type_str(self, value_model_type_str: str):
        self.value_model_type_str = value_model_type_str
        return self

    def valid(self) -> bool:
        return super().valid() and self.batch_size is not None and self.learning_rate is not None and \
                self.discount_factor is not None and self.replay_buffer_size is not None and \
                self.annealing_num_steps is not None and self.value_model_type_str is not None and \
                self.buffer_storage_type is not None and self.exploration_policy is not None and \
                self.trainer

    @abstractmethod
    def init_value_model(self):
        pass

    @abstractmethod
    def init_exploration_policy(self):
        pass

    @abstractmethod
    def init_trainer(self):
        pass
