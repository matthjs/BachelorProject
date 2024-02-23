from abc import ABC, abstractmethod

import torch
from torchrl.data import LazyMemmapStorage, ReplayBuffer


class Agent(ABC):
    """
    Agent abstract base class.
    """

    class EnvInfo:
        """
        Inner class for recording game information.
        """

        def __init__(self):
            self.done = False
            self.info = None

        def set_done(self, done):
            self.done = done

        def set_info(self, info):
            self.info = info

    def __init__(self, models, obs_space, action_space, device, memory: ReplayBuffer):
        """
        Agent Base Class constructor.
        """
        self.env_info = self.EnvInfo()
        self.replay_buffer = memory
        self.models = models
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def add_trajectory(self, trajectory):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def policy(self, state):
        pass

    @abstractmethod
    def load_parameters(self):
        pass

    @abstractmethod
    def save_parameters(self):
        pass

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

    def env_info(self) -> EnvInfo:
        return self.env_info
