from abc import ABC, abstractmethod

from builders.abstractagentbuilder import AbstractAgentBuilder
from util.fetchdevice import fetch_device


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


class AbstractAgent(ABC):
    """
    Agent abstract base class.
    """

    def __init__(self, models, state_space, action_space):
        """
        """
        self.device = fetch_device()
        self._env_info = EnvInfo()
        self._models = models
        self._state_space = state_space
        self._action_space = action_space

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

    def load_model(self, path: str):
        pass

    def save_model(self, path: str):
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

    @property
    def models(self) -> dict:
        return self._models

    @property
    def env_info(self) -> EnvInfo:
        return self._env_info
