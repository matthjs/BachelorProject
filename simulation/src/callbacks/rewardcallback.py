import os
from typing import Any

import numpy as np
import pandas as pd
from cloudpickle import cloudpickle
from loguru import logger

from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry


class RewardCallback(AbstractCallback):
    """
    Callback to track and record the reward of an agent during training.
    Optionally, save the agent when the highest average return is achieved.
    """

    def __init__(self, save_agent_on_best: bool = False) -> None:
        """
        Constructor for RewardCallback.

        :param save_agent_on_best: Whether to save the agent when the highest average return is achieved.
        """
        super().__init__()
        self.episode_reward: float = 0
        self.highest_avg_return: float = -9999999
        self.metrics_tracker = None
        self._save = save_agent_on_best

    def init_callback(self,
                      experiment_id: str,
                      mode: str,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config: Any,
                      df: pd.DataFrame,
                      metrics_tracker_registry: MetricsTrackerRegistry,
                      logging: bool = False,
                      extra: Any = None) -> None:
        """
        Initialize the callback with experiment-specific details.

        :param experiment_id: ID of the experiment.
        :param mode: Mode of the experiment ("train" or "eval").
        :param agent: Agent instance.
        :param agent_id: ID of the agent.
        :param agent_config: Configuration of the agent.
        :param df: DataFrame for logging metrics.
        :param metrics_tracker_registry: Registry for tracking metrics.
        :param logging: Whether to enable logging.
        :param extra: Additional information.
        """
        super().init_callback(experiment_id, mode, agent, agent_id, agent_config, df, metrics_tracker_registry, logging, extra)
        self.metrics_tracker = metrics_tracker_registry.get_tracker(mode)
        self.episode_reward = 0
        self.highest_avg_return = -9999999

    def _save_to_dataframe(self) -> None:
        """
        Save the average return, standard deviation of return, and max return to the DataFrame.
        """
        avg_return, avg_variance = self.metrics_tracker.latest_mean_variance("return", self.agent_id)
        self.df.loc[self.df['agent_id'] == self.agent_id, "avg return " + self.mode] = round(avg_return, 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "stdev return " + self.mode] = round(np.sqrt(avg_variance), 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "max return " + self.mode] = round(self.highest_avg_return, 3)

    def _save_agent_on_best(self, save_dir: str = "../data/saved_agents/") -> None:
        """
        Save the agent when the highest average return is achieved.

        :param save_dir: Directory to save the agent.
        """
        save_dir = os.path.join(save_dir, self.experiment_id)

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f"{self.agent_id}.pkl"), "wb") as f:
            cloudpickle.dump(self.agent, f)

    def on_step(self, action: Any, reward: float, new_obs: Any, done: bool) -> bool:
        """
        Callback for each step of the environment.

        :param action: The action taken by the agent.
        :param reward: The reward received from the environment.
        :param new_obs: The new observation from the environment.
        :param done: Whether the episode is done.
        :return: Whether to continue the experiment.
        """
        super().on_step(action, reward, new_obs, done)
        self.episode_reward += reward
        return True

    def on_episode_end(self) -> None:
        """
        Callback at the end of an episode.
        """
        super().on_episode_end()
        self.metrics_tracker.record_scalar("return", self.agent_id, self.episode_reward)

        current_return = self.metrics_tracker.value_history("return")[self.agent_id][-1]
        if current_return > self.highest_avg_return:
            self.highest_avg_return = current_return
            if self._save:
                self._save_agent_on_best()

        if self.logging and self.num_episodes % 20 == 0:
            logger.debug(f"{self.agent_id} episode reward / highest episode reward: {self.episode_reward} / {self.highest_avg_return}")

        self.episode_reward = 0

    def on_training_end(self) -> None:
        """
        Callback at the end of training.
        """
        super().on_training_end()
        mean, variance = self.metrics_tracker.latest_mean_variance("return", self.agent_id)

        if self.logging:
            print(f"Avg return - {self.agent_id} - {mean:.3f} +- {np.sqrt(variance):.3f}")

        self._save_to_dataframe()
