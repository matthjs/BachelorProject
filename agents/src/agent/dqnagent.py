import copy
from abc import ABC

import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.agent import Agent
from agent.valueagent import ValueAgent
from exploration.egreedy import EpsilonGreedy
from trainers.dqntrainer import DQNTrainer
from util.fetchdevice import fetch_device


class DQNAgent(ValueAgent):
    def __init__(self, model, exploration_policy: EpsilonGreedy, param_copying=20,
                 replay_buffer=ReplayBuffer(storage=LazyTensorStorage(10000, device=fetch_device()))):
        target = copy.deepcopy(model)   # Prob. problem with this
        super().__init__({"value_network": model, "target_network": target},
                         memory=replay_buffer,
                         exploration_policy=exploration_policy,
                         trainer=DQNTrainer(model=model,
                                            target=target,
                                            batch_size=1, buf=replay_buffer))
        self._param_copying = param_copying
        self._train_count = 0

    def update(self) -> None:
        if self._train_count == self._param_copying:
            self._train_count = 0
            self.models["target_network"].load_state_dict(self.models["value_network"].state_dict())

        super().update()
        self._train_count += 1
