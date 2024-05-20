from abc import ABC, abstractmethod

import torch


class AbstractBayesianOptimizerRL(ABC):
    """
    This is an abstract base class describing Bayesian action selection.
    """

    @abstractmethod
    def fit(self, new_train_x: torch.tensor, new_train_y: torch.tensor, hyperparameter_fitting=True) -> None:
        pass

    @abstractmethod
    def extend_dataset(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        pass

    @abstractmethod
    def dataset(self) -> tuple[torch.tensor, torch.tensor]:
        pass

    @abstractmethod
    def choose_next_action(self, state: torch.tensor) -> int:
        pass

    @abstractmethod
    def state_action_value(self, state_batch: torch.tensor, action_batch: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def max_state_action_value(self, state_batch: torch.tensor, device=None) -> tuple[torch.tensor, torch.tensor]:
        pass

    @abstractmethod
    def get_current_gp(self):
        pass
