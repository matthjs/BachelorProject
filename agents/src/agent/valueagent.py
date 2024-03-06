from abc import ABC, abstractmethod

import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.abstractagent import AbstractAgent
from builders.valueagentbuilder import ValueAgentBuilder
from exploration.egreedy import EpsilonGreedy
from trainers.rltrainer import RLTrainer
from trainers.trainer import Trainer
from util.fetchdevice import fetch_device


class ValueAgent(AbstractAgent, ABC):

    def __init__(self,
                 models,
                 state_space,
                 action_space,
                 replay_buffer_size: int,
                 buffer_storage_type=LazyTensorStorage):
        """
        """
        super().__init__(models, state_space, action_space)
        self._replay_buffer = ReplayBuffer(storage=buffer_storage_type(
                                           max_size=replay_buffer_size,
                                           device=fetch_device()))
        self._exploration_policy = None
        self._trainer = None

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the replay buffer.
        NOTE: trajectory is converted to tensor and moved to self.device.
        :param trajectory = (state, action, reward, next_state)
        """
        state, action, reward, next_state = trajectory
        state_t = torch.as_tensor(state, device=self.device)
        action_t = torch.as_tensor(action, device=self.device)
        reward_t = torch.as_tensor(reward, device=self.device)
        next_state_t = torch.as_tensor(next_state, device=self.device)
        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t))

    def update(self) -> None:
        """
        Update Q-estimate and exploration (epsilon) factor.
        Epsilon only updated upon environment end.
        :return:
        """
        if self.env_info.done:
            self._exploration_policy.step()  # Adjust epsilon
        self._trainer.train()  # Gradient update step ~ TD target.

    def policy(self, state):
        """
        Delegates action selection to the epsilon greedy class.
        :param state: of the environment.
        :return: an action.
        """
        return self._exploration_policy.action(state)