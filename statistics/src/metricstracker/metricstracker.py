import threading
from typing import List, Optional, Union, Dict, Tuple, SupportsFloat
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from util.welford import Welford


class MetricsTracker:
    """
    Thread-safe object for recording metrics. Slight abuse of the Singleton pattern similar
    to how a logging object is designed.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._loss_aggr = Welford()
            cls._reward_aggr = Welford()
            cls._loss_history: Dict[str, tuple] = defaultdict(lambda: ([], []))
            cls._reward_history: Dict[str, tuple] = defaultdict(lambda: ([], []))
        return cls._instance

    def to_csv(self, filename: str) -> None:
        """
        Write the metrics to a csv file.

        :param filename: The name of the csv file.
        """
        # Implementation for writing metrics to a CSV file
        pass

    def plot(self) -> None:
        """
        Plot the metrics to a matplotlib figure.
        """
        with self._lock:

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

            for agent_id, (mean_losses, var_losses) in self._loss_history.items():
                x_loss = np.linspace(0, len(mean_losses), len(mean_losses), endpoint=False)
                axes[0].plot(x_loss, mean_losses, label=f'{agent_id} Loss')
                axes[0].fill_between(x_loss,
                                     mean_losses - np.sqrt(var_losses) * 0.1,
                                     mean_losses + np.sqrt(var_losses) * 0.1,
                                     alpha=0.2)

            for agent_id, (mean_rewards, var_rewards) in self._reward_history.items():
                x_reward = np.linspace(0, len(mean_rewards), len(mean_rewards), endpoint=False)
                axes[1].plot(x_reward, mean_rewards, label=f'{agent_id} Loss')
                axes[1].fill_between(x_reward,
                                     mean_rewards - np.sqrt(var_rewards) * 0.1,
                                     mean_rewards + np.sqrt(var_rewards) * 0.1,
                                     alpha=0.2)

            axes[0].set_title('Loss History')
            axes[0].set_xlabel('Episodes')
            axes[0].set_ylabel('Average Loss')
            axes[0].legend()

            axes[1].set_title('Reward History')
            axes[1].set_xlabel('Episodes')
            axes[1].set_ylabel('Average Reward')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

    def record_loss(self, agent_id: str, loss: float) -> None:
        """
        Record a loss value for a specific agent.

        :param agent_id: The identifier of the agent.
        :param loss: The loss value to record.
        """
        with self._lock:
            self._loss_aggr.update_aggr(loss)
            mean_losses, variance_losses = self._loss_history[agent_id]
            mean, var = self._loss_aggr.get_curr_mean_variance()
            mean_losses.append(mean)
            variance_losses.append(var)

    def record_reward(self, agent_id: str, reward: Union[float, int, SupportsFloat]) -> None:
        """
        Record a reward value for a specific agent.

        :param agent_id: The identifier of the agent.
        :param reward: The reward value to record.
        """
        with self._lock:
            self._reward_aggr.update_aggr(reward)
            mean_rewards, variance_rewards = self._reward_history[agent_id]
            mean, var = self._reward_aggr.get_curr_mean_variance()
            mean_rewards.append(mean)
            variance_rewards.append(var)

    def latest_loss(self, agent_id: str) -> Optional[float]:
        """
        Get the latest recorded loss value for a specific agent.

        :param agent_id: The identifier of the agent.
        :return: The latest recorded loss value for the agent, or None if no loss has been recorded.
        """
        with self._lock:
            loss_values = self._loss_history.get(agent_id)
            if loss_values:
                return loss_values[-1]
            else:
                return None

    def latest_reward(self, agent_id: str) -> Optional[Union[float, int]]:
        """
        Get the latest recorded reward value for a specific agent.

        :param agent_id: The identifier of the agent.
        :return: The latest recorded reward value for the agent, or None if no reward has been recorded.
        """
        with self._lock:
            reward_values = self._reward_history.get(agent_id)
            if reward_values:
                return reward_values[-1]
            else:
                return None

    def clear(self) -> None:
        """
        Clear the recorded metrics (loss and reward history) for all agents.
        """
        with self._lock:
            self._loss_history.clear()
            self._reward_history.clear()

    @property
    def loss_history(self) -> dict[str, tuple]:
        """
        Get the history of loss values for all agents.

        :return: A dictionary containing the loss history for each agent.
        """
        with self._lock:
            return self._loss_history

    @property
    def reward_history(self) -> dict[str, tuple]:
        """
        Get the history of reward values for all agents.

        :return: A dictionary containing the reward history for each agent.
        """
        with self._lock:
            return self._reward_history