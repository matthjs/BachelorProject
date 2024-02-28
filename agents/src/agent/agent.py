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

    def __init__(self, models, memory: ReplayBuffer):
        """
        Agent Base Class constructor.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._env_info = self.EnvInfo()
        self.models = models
        self._replay_buffer = memory

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a to replay buffer.
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
    def env_info(self) -> EnvInfo:
        return self._env_info
