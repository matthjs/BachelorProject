from abc import ABC, abstractmethod

from builders.abstractagentbuilder import BaseAgentBuilder
from exploration.egreedy import EpsilonGreedy
from trainers.rltrainer import RLTrainer


class ValueAgentBuilder(BaseAgentBuilder, ABC):
    def __init__(self):
        super().__init__()
        self.exploration_policy = None
        self.trainer = None

    def set_exploration_policy(self, exploration_policy: EpsilonGreedy):
        self.exploration_policy = exploration_policy
        return self

    def set_trainer(self, trainer: RLTrainer):
        """
        Most likely you would want to instantiate this in build as you need knowledge of
        what models are used and often cannot predict what trainer is required.
        :param trainer:
        :return:
        """
        self.trainer = trainer
        return self

    def valid(self) -> bool:
        return super().valid() and self.exploration_policy is not None and self.trainer is not None

    @abstractmethod
    def init_trainer(self) -> None:
        pass
