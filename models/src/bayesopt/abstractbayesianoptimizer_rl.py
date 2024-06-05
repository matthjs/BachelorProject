from abc import ABC, abstractmethod

import torch


class AbstractBayesianOptimizerRL(ABC):
    """
    This is an abstract base class describing Bayesian action selection.
    """

    @abstractmethod
    def fit(self, new_train_x: torch.Tensor, new_train_y: torch.Tensor, hyperparameter_fitting: bool = True) -> None:
        """
        Fit the surrogate model with new training data.
        :param new_train_x: New training input data.
        :param new_train_y: New training target data.
        :param hyperparameter_fitting: Whether to fit hyperparameters. Default is True.
        """
        pass

    @abstractmethod
    def extend_dataset(self, new_train_x: torch.Tensor, new_train_y: torch.Tensor) -> None:
        """
        Extend the dataset with new training data.
        :param new_train_x: New training input data.
        :param new_train_y: New training target data.
        """
        pass

    @abstractmethod
    def dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current dataset.
        :return: A tuple containing input data and target data.
        """
        pass

    @abstractmethod
    def choose_next_action(self, state: torch.Tensor) -> int:
        """
        Choose the next action based on the current state.
        :param state: The current state.
        :return: The chosen action.
        """
        pass

    @abstractmethod
    def state_action_value(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        """
        Get the value of state-action pairs.
        :param state_batch: Batch of states.
        :param action_batch: Batch of actions.
        :return: The value of state-action pairs.
        """
        pass

    @abstractmethod
    def max_state_action_value(self, state_batch: torch.Tensor, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the maximum value and action for a given state.
        :param state_batch: Batch of states.
        :param device: Device to perform computations on (optional).
        :return: The maximum value and corresponding action.
        """
        pass

    @abstractmethod
    def get_current_gp(self):
        """
        Get the current Gaussian Process model.
        """
        pass
