from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    """
    Performs the optimization process / learning parameters
    given model and data.
    """

    @abstractmethod
    def train(self) -> None:
        pass
