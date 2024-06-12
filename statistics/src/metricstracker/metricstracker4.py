import os
import threading
from typing import List, Optional, Union, Dict, Tuple, SupportsFloat, Any
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from util.welford import Welford


class MetricsTracker4:
    """
    Thread-safe object for recording metrics.
    """

    def __init__(self):
        " map: metric_name -> {agent_type : aggregate_obj}"
        self._aggregates: Dict[str, Dict[str, Welford]] = {}
        " map: metric_name -> {agent_type: ([means], [vars])}"
        self._aggregates_history: Dict[str, Dict[str, tuple]] = {}
        self._values_history: Dict[str, Dict[str, List[float]]] = {}

    def to_csv(self, filename: str) -> None:
        """
        Write the metrics to a csv file.

        :param filename: The name of the csv file.
        """
        # Implementation for writing metrics to a CSV file
        pass

    def register_metric(self, metric_name: str) -> None:
        self._aggregates[metric_name] = {}
        self._values_history[metric_name] = {}
        self._aggregates_history[metric_name] = defaultdict(lambda: ([], []))

    def plot_metric(self, metric_name, plot_path="./", x_axis_label="Episodes", y_axis_label='Average',
                    title="History") -> None:
        """
        Plot the metrics to a matplotlib figure.
        """
        if metric_name not in self._aggregates:
            raise AttributeError(f"Metric name {metric_name} not registered")

        fig, ax = plt.subplots(figsize=(10, 8))

        for agent_id, (mean_returns, var_returns) in self._aggregates_history[metric_name].items():
            x_return = np.linspace(0, len(mean_returns), len(mean_returns), endpoint=False)
            ax.plot(x_return, mean_returns, label=f'{agent_id} agent')
            ax.fill_between(x_return,
                                mean_returns - np.sqrt(var_returns) * 0.1,
                                mean_returns + np.sqrt(var_returns) * 0.1,
                                alpha=0.2)

        ax.set_title(metric_name + " " + title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label + " " + metric_name)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        # Create directory if it does not exist
        plot_dir = '../plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(plot_path)

        plt.show()

    def record_scalar(self, metric_name: str, agent_id: str, val: float) -> None:
        """
        Record val
        :param metric_name:
        :param agent_id:
        :param val:
        """
        if metric_name not in self._aggregates:
            raise AttributeError(f"Metric name {metric_name} not registered")
        if agent_id not in self._aggregates[metric_name]:
            self._aggregates[metric_name][agent_id] = Welford()
            self._values_history[metric_name][agent_id] = []

        # Update mean, var estimate
        self._aggregates[metric_name][agent_id].update_aggr(val)
        self._values_history[metric_name][agent_id].append(val)
        means, variances = self._aggregates_history[metric_name][agent_id]
        mean, variance = self._aggregates[metric_name][agent_id].get_curr_mean_variance()
        means.append(mean)
        variances.append(variance)

    def latest_mean_variance(self, metric_name: str, agent_id: str) -> tuple[Any, Any] | None:
        """
        Get latest mean variance estimate.
        :param metric_name:
        :param agent_id:
        :return:
        """
        means, variances = self._aggregates_history[metric_name].get(agent_id)
        if means:
            return means[-1], variances[-1]
        else:
            return None

    def metric_history(self, metric_name: str) -> dict[str, tuple]:
        return self._aggregates_history[metric_name]

    def value_history(self, metric_name: str) -> dict[str, List[float]]:
        return self._values_history[metric_name]