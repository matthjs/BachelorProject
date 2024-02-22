from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    Performs the optimization process / learning parameters
    given model and input
    """

    @abstractmethod
    def train(self) -> None:
        pass
