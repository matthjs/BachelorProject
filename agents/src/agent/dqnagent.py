import copy
from abc import ABC

import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.abstractagent import Agent
from agent.valueagent import ValueAgent
from exploration.egreedy import EpsilonGreedy
from trainers.dqntrainer import DQNTrainer
from util.fetchdevice import fetch_device


class DQNAgent(ValueAgent):
    def __init__(self, models: dict,
                 exploration_policy: EpsilonGreedy,
                 param_copying=20,
                 batch_size=32,
                 discount_factor=0.9,
                 replay_buffer_size=10000):
        super().__init__(models=models,
                         memory=replay_buffer,
                         exploration_policy=exploration_policy,
                         trainer=DQNTrainer(model=models["value_network"],
                                            target=models["target_network"],
                                            batch_size=1, buf=replay_buffer))
        self._param_copying = param_copying
        self._train_count = 0

    def __init__(self, builder:):
        pass

    def update(self) -> None:
        if self._train_count == self._param_copying:
            self._train_count = 0
            self.models["target_network"].load_state_dict(self.models["value_network"].state_dict())

        super().update()
        self._train_count += 1
