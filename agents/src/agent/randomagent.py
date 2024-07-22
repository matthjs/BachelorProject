from agent.abstractagent import AbstractAgent
import gymnasium as gym
import numpy as np


class RandomAgent(AbstractAgent):
    """
    A simple agent that takes random actions.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Constructor for RandomAgent.

        :param env: The environment.
        """
        super().__init__({}, env.observation_space, env.action_space)
        self.action_space = env.action_space

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the agent (not used in RandomAgent).

        :param trajectory: Trajectory tuple.
        """
        pass

    def update(self) -> None:
        """
        Update the agent (not used in RandomAgent).
        """
        pass

    def policy(self, state: np.ndarray) -> int:
        """
        Get the policy for the given state, which is a random action in this case.

        :param state: The state.
        :return: A random action.
        """
        return self.action_space.sample()
