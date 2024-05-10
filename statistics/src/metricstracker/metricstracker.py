import os
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
            cls._loss_aggr: Dict[str, Welford] = {}
            cls._return_aggr: Dict[str, Welford] = {}
            cls._loss_history: Dict[str, tuple] = defaultdict(lambda: ([], []))
            cls._return_history: Dict[str, tuple] = defaultdict(lambda: ([], []))
        return cls._instance

    def to_csv(self, filename: str) -> None:
        """
        Write the metrics to a csv file.

        :param filename: The name of the csv file.
        """
        # Implementation for writing metrics to a CSV file
        pass

    def plot(self, name="plot") -> None:
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

            for agent_id, (mean_returns, var_returns) in self._return_history.items():
                x_return = np.linspace(0, len(mean_returns), len(mean_returns), endpoint=False)
                axes[1].plot(x_return, mean_returns, label=f'{agent_id} Returns')
                axes[1].fill_between(x_return,
                                     mean_returns - np.sqrt(var_returns) * 0.1,
                                     mean_returns + np.sqrt(var_returns) * 0.1,
                                     alpha=0.2)

            axes[0].set_title('Loss History')
            axes[0].set_xlabel('Time steps')
            axes[0].set_ylabel('Average Loss')
            axes[0].legend()

            axes[1].set_title('return History')
            axes[1].set_xlabel('Episodes')
            axes[1].set_ylabel('Average return')
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            plt.savefig("../plots/" + name + ".png")

    def plot_return(self, x_axis_label="Episodes", y_axis_label='Average Return', title="Return History") -> None:
        """
        Plot the metrics to a matplotlib figure.
        """
        with self._lock:
            fig, ax = plt.subplots(figsize=(10, 8))

            for agent_id, (mean_returns, var_returns) in self._return_history.items():
                x_return = np.linspace(0, len(mean_returns), len(mean_returns), endpoint=False)
                ax.plot(x_return, mean_returns, label=f'{agent_id} agent')
                ax.fill_between(x_return,
                                mean_returns - np.sqrt(var_returns) * 0.1,
                                mean_returns + np.sqrt(var_returns) * 0.1,
                                alpha=0.2)

            ax.set_title(title)
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()
            # Create directory if it does not exist
            plot_dir = '../plots'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.savefig('../plots/result.png')

    def record_loss(self, agent_id: str, loss: float) -> None:
        """
        Record a loss value for a specific agent.

        :param agent_id: The identifier of the agent.
        :param loss: The loss value to record.
        """
        with self._lock:
            if agent_id not in self._loss_aggr:
                self._loss_aggr[agent_id] = Welford()

            self._loss_aggr[agent_id].update_aggr(loss)
            mean_losses, variance_losses = self._loss_history[agent_id]
            mean, var = self._loss_aggr[agent_id].get_curr_mean_variance()
            mean_losses.append(mean)
            variance_losses.append(var)

    def record_return(self, agent_id: str, return_val: Union[float, int, SupportsFloat]) -> None:
        """
        Record a return value for a specific agent.
        NOTE: Code duplication == bad.

        :param agent_id: The identifier of the agent.
        :param return_val: The return value to record.
        """
        with self._lock:
            if agent_id not in self._return_aggr:
                self._return_aggr[agent_id] = Welford()
            self._return_aggr[agent_id].update_aggr(return_val)
            mean_returns, variance_returns = self._return_history[agent_id]
            mean, var = self._return_aggr[agent_id].get_curr_mean_variance()
            mean_returns.append(mean)
            variance_returns.append(var)

    def latest_average_return(self, agent_id: str) -> Optional[float]:
        """
        Get the latest recorded average return value for a specific agent.

        :param agent_id: The identifier of the agent.
        :return: The latest recorded average return value for the agent, or None if no return has been recorded.
        """
        with self._lock:
            return_values, _ = self._return_history.get(agent_id)
            if return_values:
                return return_values[-1]
            else:
                return None

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

    def clear(self) -> None:
        """
        Clear the recorded metrics (loss and return history) for all agents.
        """
        with self._lock:
            self._loss_history.clear()
            self._return_history.clear()

    @property
    def loss_history(self) -> dict[str, tuple]:
        """
        Get the history of loss values for all agents.

        :return: A dictionary containing the loss history for each agent.
        """
        with self._lock:
            return self._loss_history

    @property
    def return_history(self) -> dict[str, tuple]:
        """
        Get the history of return values for all agents.

        :return: A dictionary containing the return history for each agent.
        """
        with self._lock:
            return self._return_history