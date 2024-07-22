import pandas as pd
from typing import Any
from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry

class EarlyStopCallback(AbstractCallback):
    """
    Callback to stop training early if a reward threshold is reached.
    NOTE: This does not work with Stable Baselines agents.
    """

    def __init__(self, reward_callback: AbstractCallback, reward_threshold: float,
                 threshold_reach_count: int = 1) -> None:
        """
        Constructor for EarlyStopCallback.

        :param reward_callback: Callback to track rewards.
        :param reward_threshold: Reward threshold to stop training.
        :param threshold_reach_count: Number of times the reward threshold must be reached before stopping.
        """
        super().__init__()
        self.reward_callback = reward_callback
        self.reward_threshold = reward_threshold
        self.threshold_reach_count_init = threshold_reach_count
        self.threshold_reach_count = threshold_reach_count
        self.metrics_tracker = None
        self.episode_reward = 0
        self.threshold_reached = False

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
        super().init_callback(experiment_id, mode, agent, agent_id, agent_config, df, metrics_tracker_registry, logging,
                              extra)
        self.metrics_tracker = metrics_tracker_registry.get_tracker(mode)
        self.reward_callback.init_callback(experiment_id, mode, agent, agent_id, agent_config, df,
                                           metrics_tracker_registry, logging, extra)

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
        self.reward_callback.on_step(action, reward, new_obs, done)
        return True

    def on_episode_end(self) -> None:
        """
        Callback at the end of an episode.
        """
        super().on_episode_end()
        self.reward_callback.on_episode_end()
        if not self.threshold_reached:
            current_return = self.metrics_tracker.value_history("return")[self.agent_id][-1]
            if current_return >= self.reward_threshold:
                self.threshold_reach_count -= 1
                if self.threshold_reach_count <= 0:
                    self.agent.disable_training()
                    self.threshold_reached = True
            else:
                self.threshold_reach_count = self.threshold_reach_count_init

    def on_training_end(self) -> None:
        """
        Callback at the end of training.
        """
        super().on_training_end()
        self.reward_callback.on_training_end()
