from abc import ABC
from typing import Any, Optional

import pandas as pd
from agent.abstractagent import AbstractAgent
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry


class AbstractCallback(ABC):
    """
    Abstract base class for callbacks in reinforcement learning experiments.
    """

    def __init__(self) -> None:
        """
        Constructor for AbstractCallback.
        """
        self.mode: Optional[str] = None
        self.num_steps: int = 0
        self.num_episodes: int = 0
        self.logging: bool = False
        self.extra: Optional[Any] = None
        self.agent: Optional[AbstractAgent] = None
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[Any] = None
        self.metrics_tracker_registry: Optional[Any] = None
        self.df: Optional[pd.DataFrame] = None
        self.experiment_id: Optional[str] = None
        self.old_obs: Optional[Any] = None

    def init_callback(self,
                      experiment_id: str,
                      mode: str,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config: Any,
                      df: pd.DataFrame,
                      metrics_tracker_registry: MetricsTrackerRegistry,
                      logging: bool = False,
                      extra: Optional[Any] = None) -> None:
        """
        Initialize the callback with experiment-specific details.

        :param experiment_id: ID of the experiment.
        :param mode: Mode of the experiment ("train" or "eval").
        :param agent: Agent instance.
        :param agent_id: ID of the agent.
        :param agent_config: Configuration of the agent.
        :param df: DataFrame for logging metrics.
        :param metrics_tracker_registry: Registry for metric trackers.
        :param logging: Whether to enable logging.
        :param extra: Additional information.
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode {mode}.")
        self.mode = mode

        self.experiment_id = experiment_id
        self.agent = agent
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.df = df
        self.metrics_tracker_registry = metrics_tracker_registry
        self.extra = extra
        self.old_obs = None
        self.logging = logging

    def on_step(self, action: Any, reward: float, new_obs: Any, done: bool) -> bool:
        """
        Callback for each step of the environment.

        :param action: The action taken by the agent.
        :param reward: The reward received from the environment.
        :param new_obs: The new observation from the environment.
        :param done: Whether the episode is done.
        :return: Whether to continue the run.
        """
        self.num_steps += 1
        self.old_obs = new_obs
        return True

    def on_episode_end(self) -> None:
        """
        Callback at the end of an episode.
        """
        self.num_episodes += 1

    def on_training_start(self) -> None:
        """
        Callback at the start of training.
        """
        pass

    def on_training_end(self) -> None:
        """
        Callback at the end of training.
        """
        pass

    def on_update_start(self) -> None:
        """
        Callback at the start of an update.
        """
        pass

    def on_update_end(self) -> None:
        """
        Callback at the end of an update.
        """
        pass
