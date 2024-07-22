import os
from typing import List, Optional, Dict, Any
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from welford.welford import Welford


class MetricsTracker:
    """
    NOT Thread-safe object for recording metrics.
    """

    def __init__(self):
        " map: metric_name -> {agent_type : aggregate_obj}"
        self._aggregates: Dict[str, Dict[str, Welford]] = {}
        " map: metric_name -> {agent_type: ([means], [vars])}"
        self._aggregates_history: Dict[str, Dict[str, tuple]] = {}
        self._values_history: Dict[str, Dict[str, List[float]]] = {}

    def register_metric(self, metric_name: str) -> None:
        """
        Register a new metric to be tracked.

        :param metric_name: The name of the metric to register.
        """
        self._aggregates[metric_name] = {}
        self._values_history[metric_name] = {}
        self._aggregates_history[metric_name] = defaultdict(lambda: ([], []))

    def plot_metric(self, metric_name, plot_path="./", x_axis_label="Episodes", y_axis_label='Average',
                    title="History", figsize=(10, 6), fontsize=14, linewidth=1.5,
                    id_list: Optional[List[str]] = None,
                    color_list: Optional[dict] = None) -> None:
        """
        Plot the metrics to a matplotlib figure.

        :param metric_name: The name of the metric to plot.
        :param plot_path: The path to save the plot. Default is "./".
        :param x_axis_label: The label for the x-axis. Default is "Episodes".
        :param y_axis_label: The label for the y-axis. Default is "Average".
        :param title: The title of the plot. Default is "History".
        :param figsize: The size of the figure. Default is (10, 6).
        :param fontsize: The font size for labels and title. Default is 14.
        :param linewidth: The width of the lines in the plot. Default is 1.5.
        :param id_list: Optional list of agent IDs to include in the plot. Default is None.
        :param color_list: Optional dictionary of colors for the agents. Default is None.
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

        # Create directory if it does not exist
        plot_dir = '../plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(plot_path, dpi=300)

        plt.show()

    def record_scalar(self, metric_name: str, agent_id: str, val: float) -> None:
        """
        Record a scalar value for a specific metric and agent.

        :param metric_name: The name of the metric.
        :param agent_id: The ID of the agent.
        :param val: The value to record.
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
        Get the latest mean and variance estimate for a specific metric and agent.

        :param metric_name: The name of the metric.
        :param agent_id: The ID of the agent.
        :return: A tuple containing the latest mean and variance, or None if no data is available.
        """
        means, variances = self._aggregates_history[metric_name].get(agent_id)
        if means:
            return means[-1], variances[-1]
        else:
            return None

    def metric_history(self, metric_name: str) -> dict[str, tuple]:
        """
        Get the history of aggregates for a specific metric.

        :param metric_name: The name of the metric.
        :return: A dictionary mapping agent IDs to tuples of (mean history, variance history).
        """
        return self._aggregates_history[metric_name]

    def value_history(self, metric_name: str) -> dict[str, List[float]]:
        """
        Get the history of recorded values for a specific metric.

        :param metric_name: The name of the metric.
        :return: A dictionary mapping agent IDs to lists of recorded values.
        """
        return self._values_history[metric_name]
