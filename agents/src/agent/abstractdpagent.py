from abc import ABC, abstractmethod

import gymnasium as gym

from agent.abstractagent import AbstractAgent


class AbstractDPAgent(AbstractAgent, ABC):
    def __init__(self, builder):
        super().__init__(builder)
        self._gym_env = builder.gym_env

    @abstractmethod
    def iterate(self):
        pass

    def update(self):
        """
        For dynamic programming agents the update method simply runs the iterate method
        which is the function that executes the dynamic programming algorithm.
        :return:
        """
        self.iterate()
