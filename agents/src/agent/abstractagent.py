from abc import ABC, abstractmethod

import torch
from torchrl.data import LazyMemmapStorage, ReplayBuffer

from builders.abstractagentbuilder import BaseAgentBuilder, EnvInfo
from util.fetchdevice import fetch_device


class AbstractAgent(ABC):
    """
    Agent abstract base class.
    """
    def __init__(self, builder: BaseAgentBuilder):
        """
        NOTE: This constructor should not be called directly. It is
        indirectly called by BaseAgentBuilder.
        """
        self.device = builder.device
        self._env_info = builder.env_info
        self._models = builder.models
        self._replay_buffer = builder.replay_buffer

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

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def policy(self, state):
        pass

    def load_parameters(self):
        for model, model_name in enumerate(self.models):
            model.load()

    def save_parameters(self):
        for model, model_name in enumerate(self.models):
            model.save()

    def record_env_info(self, info, done=False) -> None:
        """
        Necessary for monte carlo agents
        to use the same interface as TD agents.
        Recording "done" can be used to prevent
        the agent from training until an episode
        is finished.
        """
        self.env_info.set_done(done)
        self.env_info.set_info(info)

    @property
    def models(self) -> dict:
        return self._models

    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer\

    @property
    def env_info(self) -> EnvInfo:
        return self._env_info
