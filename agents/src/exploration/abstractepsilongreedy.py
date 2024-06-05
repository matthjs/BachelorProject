from abc import ABC, abstractmethod


# ! Not used !


class AbstractEpsilonGreedy(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def action(self, state):
        pass
