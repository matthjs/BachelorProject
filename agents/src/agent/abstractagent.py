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
        self._state_space = builder.state_space
        self._action_space = builder.action_space

    @abstractmethod
    def add_trajectory(self, trajectory: tuple) -> None:
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def policy(self, state):
        pass

    # def load_parameters(self):
    #    for model, model_name in enumerate(self.models):
    #        model.load()

    # def save_parameters(self):
    #     for model, model_name in enumerate(self.models):
    #         model.save()

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
    def env_info(self) -> EnvInfo:
        return self._env_info
