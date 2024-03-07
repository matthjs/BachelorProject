from abc import ABC, abstractmethod

from agent.abstractagent import AbstractAgent


class AbstractDPAgent(AbstractAgent, ABC):
    def __init__(self, models, state_space, action_space):
        super().__init__(models, state_space, action_space)

    @abstractmethod
    def iterate(self):
        pass

    def add_trajectory(self, trajectory: tuple) -> None:
        pass

    def update(self):
        """
        For dynamic programming agents the update method simply runs the iterate method
        which is the function that executes the dynamic programming algorithm.
        :return:
        """
        self.iterate()
