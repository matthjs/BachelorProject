import pandas as pd
from loguru import logger

from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback


class RewardCallback(AbstractCallback):
    def __init__(self):
        super().__init__()
        self.episode_reward = 0
        self.highest_avg_return = -9999999
        self.metrics_tracker = None
        
    def init_callback(self,
                      mode: str,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config,
                      df: pd.DataFrame,
                      metrics_tracker_registry,
                      verbose=0,
                      extra=None):
        super().init_callback(mode, agent, agent_id, agent_config, df, metrics_tracker_registry, verbose, extra)
        self.metrics_tracker = metrics_tracker_registry.get_tracker(mode)

    def on_step(self, action, reward, new_obs, done) -> bool:
        super().on_step(action, reward, new_obs, done)

        self.episode_reward += reward

        return True

    def on_episode_end(self) -> None:
        super().on_episode_end()
        self.metrics_tracker("return", self.agent_id, self.episode_reward)

        current_avg_return, _ = self.metrics_tracker.latest_mean_variance("return", self.agent_id)
        if current_avg_return > self.highest_avg_return:
            self.highest_average_return = current_avg_return
            # Do something maybe

        if self.verbose > 0:
            logger.debug(f"Episode reward / highest episode reward"
                         f": {self.episode_reward} / {self.highest_avg_return}")

        self.episode_reward = 0

    def on_training_end(self) -> None:
        """
        You can plot here, but this is preferably done in the simulator class.
        :return:
        """
        pass

