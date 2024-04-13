from abc import ABC, abstractmethod


class AbstractEpsilonGreedy(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def action(self, state):
        pass
