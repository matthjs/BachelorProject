from abc import ABC, abstractmethod
from typing import Any
from util.fetchdevice import fetch_device


class EnvInfo:
    """
    Inner class for recording game information.
    This is not really used by any class though...
    """

    def __init__(self) -> None:
        self.done: bool = False
        self.info: Any = None

    def set_done(self, done: bool) -> None:
        self.done = done

    def set_info(self, info: Any) -> None:
        self.info = info


class AbstractAgent(ABC):
    """
    Agent abstract base class.
    """

    def __init__(self, models: dict, state_space: Any, action_space: Any) -> None:
        """
        Constructor.
        :param models: Dictionary of models.
        :param state_space: The state space of the environment.
        :param action_space: The action space of the environment.
        """
        self.device = fetch_device()
        self._env_info = EnvInfo()  # This is not used.
        self._models = models       # This is not really used.
        self._state_space = state_space
        self._action_space = action_space

    @abstractmethod
    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Abstract method to add trajectory.
        :param trajectory: Trajectory tuple.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Abstract method to update agent.
        """
        pass

    @abstractmethod
    def policy(self, state: Any) -> Any:
        """
        Abstract method for policy.
        :param state: State of the environment.
        :return: Action.
        """
        pass

    @staticmethod
    def load_model(path: str) -> None:
        """
        Static method to load model.
        :param path: Path to the model.
        """
        raise NotImplementedError("Not implemented.")

    def save_model(self, path: str) -> None:
        """
        Method to save model.
        :param path: Path to save the model.
        """
        raise NotImplementedError("Not implemented.")

    def is_stable_baselines_wrapper(self) -> bool:
        """
        Check if the agent is a Stable Baselines wrapper.
        :return: True if it's a Stable Baselines wrapper, False otherwise.
        """
        return False

    def stable_baselines_unwrapped(self) -> None:
        """
        Get the unwrapped version of the agent if it's a Stable Baselines wrapper.
        """
        return None

    def latest_loss(self) -> float:
        raise NotImplementedError("Not implemented.")

    def record_env_info(self, info: Any, done: bool = False) -> None:
        """
        Record environment information.
        Necessary for Monte Carlo agents
        to use the same interface as TD agents.
        Recording "done" can be used to prevent
        the agent from training until an episode
        is finished.
        :param info: Information from the environment.
        :param done: Flag indicating if the episode is finished.
        """
        self.env_info.set_done(done)
        self.env_info.set_info(info)

    @property
    def models(self) -> dict:
        """
        Get the models.
        :return: Dictionary of models.
        """
        return self._models

    @property
    def env_info(self) -> EnvInfo:
        """
        Get the environment information.
        :return: Environment information object.
        """
        return self._env_info

    def hyperparameters(self) -> dict:
        """
        Get the hyperparameters.
        :return: Dictionary of hyperparameters.
        """
        return {}

    def updatable(self) -> bool:
        """
        Check if the agent is updatable.
        :return: True if the agent is updatable, False otherwise.
        """
        return True

