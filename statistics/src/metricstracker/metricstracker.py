from typing import List, Optional, Union


class MetricsTracker:
    """
    A class for tracking metrics such as loss and rewards in a Reinforcement Learning agent.

    Attributes:
        _loss_history (List[float]): A list to store the history of loss values.
        _reward_history (List[Union[float, int]]): A list to store the history of reward values.
    """

    def __init__(self):
        """
        Initialize the MetricsTracker.
        """
        self._loss_history: List[float] = []
        self._reward_history: List[Union[float, int]] = []

    @property
    def loss_history(self) -> List[float]:
        """
        Get the history of loss values.

        :return: The list of loss values recorded.
        """
        return self._loss_history

    @property
    def reward_history(self) -> List[Union[float, int]]:
        """
        Get the history of reward values.

        :return: The list of reward values recorded.
        """
        return self._reward_history

    def record_loss(self, loss: float) -> None:
        """
        Record a loss value.

        :param loss: The loss value to record.
        """
        self._loss_history.append(loss)

    def record_reward(self, reward: Union[float, int]) -> None:
        """
        Record a reward value.

        :param reward: The reward value to record.
        """
        self._reward_history.append(reward)

    @property
    def latest_loss(self) -> Optional[float]:
        """
        Get the latest recorded loss value.

        :return: The latest recorded loss value, or None if no loss has been recorded.
        """
        if self._loss_history:
            return self._loss_history[-1]
        else:
            return None

    @property
    def latest_reward(self) -> Optional[Union[float, int]]:
        """
        Get the latest recorded reward value.

        :return: The latest recorded reward value, or None if no reward has been recorded.
        """
        if self._reward_history:
            return self._reward_history[-1]
        else:
            return None

    def clear(self) -> None:
        """
        Clear the recorded metrics (loss and reward history).
        """
        self._loss_history.clear()
        self._reward_history.clear()
