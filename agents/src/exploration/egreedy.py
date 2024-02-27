import random

import gymnasium as gym
import numpy as np
import torch

from util.fetchdevice import fetch_device


def process_state(state):
    return torch.from_numpy(state).to(device=fetch_device())


class EpsilonGreedy:
    """
    Class supporting epsilon greedy action selection with decay.
    TODO: Create ABC for exploration modules.
    NOTE: This class is currently only compatible with discrete action spaces.
    NOTE: Constructor assumes gym.spaces.Discrete as type of action space but this
    requirement can be relaxed probably.
    TODO: Find some way to generalize environment specifications.
    TODO: Look into EGREEDYMODULE from torchrl.
    """
    def __init__(self, model, action_space: gym.Space, eps_init=1.0, eps_end=0.1, annealing_num_steps=5000):
        """
        :param model: action-value function estimate.
        :param action_space:
        :param eps_init: initial epsilon value. default: 1.0
        :param eps_end: final epsilon value. default: 0.1
        :param annealing_num_steps: number of steps it will take for epsilon to reach the eps_end value.
        Defaults to 1000.
        """
        self._model = model
        self._action_space = action_space
        self._epsilon = eps_init
        self._epsilon_target = eps_end
        self._epsilon_delta = (1 - self._epsilon_target) / annealing_num_steps

    def step(self) -> None:
        """
        Updates epsilon schedule [linear]
        """
        if self._epsilon > self._epsilon_target:
            self._epsilon = max(self._epsilon_target, self._epsilon - self._epsilon_delta)

    def action(self, state) -> int | float:
        """
        Limited epsilon greedy action selection.
        :param state:
        :return: action
        """
        if np.random.uniform(0, 100) >= self._epsilon * 100:
            model_output = self._model(process_state(state))
            return torch.argmax(model_output).item()

        return self._action_space.sample()
