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
                    title="History", figsize=(10, 6), fontsize=14, linewidth=1.5,
                    id_list: Optional[List[str]] = None,
                    color_list: Optional[dict] = None) -> None:
        """
        Plot the metrics to a matplotlib figure.
        """
        if metric_name not in self._aggregates:
            print(f"Metric name {metric_name} not registered")
            return

        fig, ax = plt.subplots(figsize=figsize)

        use_colors = color_list is not None and id_list is not None and len(color_list) == len(id_list)

        for agent_id, (mean_returns, var_returns) in self._aggregates_history[metric_name].items():
            if id_list is not None and agent_id not in id_list:
                continue

            color = color_list[agent_id] if use_colors else None

            if agent_id == "gpq_agent_3":
                agent_id = "GPQ (SVGP)"
            elif agent_id == "sb_dqn_1":
                continue
                agent_id = "DQN (MLP)"
            elif agent_id == "GPQ2 (DGP)":
                agent_id = "GPQ (DGP)"
            elif agent_id == "sb_dqn_2":
                agent_id = "DQN (Linear)"
            elif agent_id == "GPSARSA (DGP)":
                continue
            elif agent_id == "random":
                agent_id = "RANDOM"

            x_return = np.linspace(0, len(mean_returns), len(mean_returns), endpoint=False)
            ax.plot(x_return, mean_returns, linewidth=linewidth, label=f'{agent_id} agent', color=color)
            ax.fill_between(x_return,
                            mean_returns - np.sqrt(var_returns) * 0.1,
                            mean_returns + np.sqrt(var_returns) * 0.1,
                            alpha=0.2,
                            color=color)

        ax.set_title(metric_name + " " + title, fontsize=fontsize)
        ax.set_xlabel(x_axis_label, fontsize=fontsize)
        ax.set_ylabel(y_axis_label + " " + metric_name, fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.grid(True)

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()
        # -400, 150 for Lunar Lander
        # if metric_name != "loss":
        #    plt.ylim((-400, 150))

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
