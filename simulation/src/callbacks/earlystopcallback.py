import os

import numpy as np
import pandas as pd
from cloudpickle import cloudpickle
from loguru import logger

from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback


class EarlyStopCallback(AbstractCallback):
    # NOTE DOES NOT WORK ON STABLE BASELINES AGENTS.
    def __init__(self, reward_callback, reward_threshold, threshold_reach_count=1):
        super().__init__()
        self.reward_callback = reward_callback
        self.reward_threshold = reward_threshold
        self.threshold_reach_count = threshold_reach_count
        self.metrics_tracker = None
        self.episode_reward = 0
        self.threshold_reached = False

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
        super().init_callback(experiment_id, mode, agent, agent_id, agent_config, df, metrics_tracker_registry, logging,
                              extra)
        self.metrics_tracker = metrics_tracker_registry.get_tracker(mode)
        self.reward_callback.init_callback(experiment_id,
                                           mode, agent, agent_id, agent_config, df, metrics_tracker_registry, logging,
                                           extra)

    def on_step(self, action, reward, new_obs, done) -> bool:
        super().on_step(action, reward, new_obs, done)
        self.reward_callback.on_step(action, reward, new_obs, done)

        return True

    def on_episode_end(self) -> None:
        super().on_episode_end()
        self.reward_callback.on_episode_end()
        if not self.threshold_reached:
            current_return = self.metrics_tracker.value_history("return")[self.agent_id][-1]
            if current_return >= self.reward_threshold:
                self.threshold_reach_count -= 1
                if self.threshold_reach_count <= 0:
                    self.agent.disable_training()
                    self.threshold_reached = True
