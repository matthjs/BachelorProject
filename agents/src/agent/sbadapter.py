import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from agent.abstractagent import AbstractAgent


class StableBaselinesAdapter(AbstractAgent):
    """
    Adapter class to use Stable Baselines models as agents.
    """

    def __init__(self, model: BaseAlgorithm) -> None:
        """
        Constructor for StableBaselinesAdapter.

        :param model: Stable Baselines model to adapt.
        """
        super().__init__({}, model.observation_space, model.action_space)
        self._sb_model = model

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the agent (not used in this adapter).

        :param trajectory: Tuple containing (state, action, reward, next_state).
        """
        pass

    def update(self) -> None:
        """
        Update method (not used in this adapter).
        """
        pass

    def policy(self, state) -> int | np.ndarray:
        """
        Get the action from the Stable Baselines model.

        :param state: The state.
        :return: The action predicted by the model.
        """
        return self._sb_model.predict(state)[0]

    def is_stable_baselines_wrapper(self) -> bool:
        """
        Check if the agent is a wrapper for a Stable Baselines model.

        :return: True if it is, False otherwise.
        """
        return True

    def stable_baselines_unwrapped(self) -> BaseAlgorithm:
        """
        Get the unwrapped Stable Baselines model.

        :return: The unwrapped model.
        """
        return self._sb_model
