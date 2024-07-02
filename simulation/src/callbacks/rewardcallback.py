import os

import numpy as np
import pandas as pd
from cloudpickle import cloudpickle
from loguru import logger

from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback


class RewardCallback(AbstractCallback):
    def __init__(self, save_agent_on_best=False):
        super().__init__()
        self.episode_reward = 0
        self.highest_avg_return = -9999999
        self.metrics_tracker = None
        self._save = save_agent_on_best
        
    def init_callback(self,
                      experiment_id: str,
                      mode: str,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config,
                      df: pd.DataFrame,
                      metrics_tracker_registry,
                      logging=False,
                      extra=None):
        super().init_callback(experiment_id, mode, agent, agent_id, agent_config, df, metrics_tracker_registry, logging, extra)
        self.metrics_tracker = metrics_tracker_registry.get_tracker(mode)
        self.episode_reward = 0
        self.highest_avg_return = -9999999

    def _save_to_dataframe(self):
        avg_return, avg_variance = self.metrics_tracker.latest_mean_variance("return", self.agent_id)
        self.df.loc[self.df['agent_id'] == self.agent_id, "avg return " + self.mode] = round(avg_return, 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "stdev return " + self.mode] = round(np.sqrt(avg_variance), 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "max return " + self.mode] = round(self.highest_avg_return, 3)

    def _save_agent_on_best(self, save_dir="../data/saved_agents/"):
        save_dir = save_dir + self.experiment_id + "/"

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        with open(save_dir + self.agent_id + ".pkl", "wb") as f:
            cloudpickle.dump(self.agent, f)

    def on_step(self, action, reward, new_obs, done) -> bool:
        super().on_step(action, reward, new_obs, done)

        self.episode_reward += reward

        return True

    def on_episode_end(self) -> None:
        super().on_episode_end()
        self.metrics_tracker.record_scalar("return", self.agent_id, self.episode_reward)

        # current_avg_return, _ = self.metrics_tracker.latest_mean_variance("return", self.agent_id)
        current_return = self.metrics_tracker.value_history("return")[self.agent_id][-1]
        if current_return > self.highest_avg_return:
            self.highest_avg_return = current_return
            if self._save:
                self._save_agent_on_best()

        if self.logging and self.num_episodes % 20 == 0:
            logger.debug(f"{self.agent_id } episode reward / highest episode reward"
                         f": {self.episode_reward} / {self.highest_avg_return}")

        self.episode_reward = 0

    def on_training_end(self) -> None:
        """
        You can plot here, but this is preferably done in the simulator class.
        :return:
        """
        mean, variance = self.metrics_tracker.latest_mean_variance("return", self.agent_id)

        if self.logging:
            print(f"Avg return - {self.agent_id} - {mean:.3f} +- {np.sqrt(variance):.3f}")

        self._save_to_dataframe()
