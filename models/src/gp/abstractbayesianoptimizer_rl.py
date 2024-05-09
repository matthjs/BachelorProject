from abc import ABC, abstractmethod
from botorch.models.model import Model


class AbstractBayesianOptimizerRL(ABC):
    """
    This is an abstract base class describing Bayesian action selection.
    """
    @abstractmethod
    def fit(self, new_train_x, new_train_y, hyperparameter_fitting=True) -> Model:
        """
        Fit a new (Gaussian process) model to the additional data.
        :param new_train_x:
        :param new_train_y:
        :param hyperparameter_fitting:
        :return:
        """
        pass

    @abstractmethod
    def extend_dataset(self, new_train_x, new_train_y):
        pass

    @abstractmethod
    def choose_next_action(self, state):
        """
        Choose an action from the model.
        :param state:
        :return: an action.
        """
        pass

    @abstractmethod
    def max_state_action_value(self, state_batch, device=None):
        pass
