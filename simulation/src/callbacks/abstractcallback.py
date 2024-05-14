from abc import ABC, abstractmethod

import pandas as pd

from agent.abstractagent import AbstractAgent
from callbacks.attributewrapper import AttributeWrapper

class AbstractCallback(ABC):
    def __init__(self, mode, verbose: int = 0):
        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode {mode}.")

        self.mode = mode
        self.num_steps = 0  # type: int
        self.num_episodes = 0
        self.verbose = verbose
        self.extra = None
        self.agent = None
        self.agent_id = None
        self.agent_config = None
        self.metrics_tracker_registry = None
        self.df = None

    def init_callback(self,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config,
                      df: pd.DataFrame,
                      metrics_tracker_registry,
                      extra=None):
        self.data_map = {}
        self.agent = agent
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.df = df
        self.metrics_tracker_registry = metrics_tracker_registry
        self.extra = extra
        self.old_obs = None

    def on_step(self, action, reward, new_obs, done) -> bool:
        self.num_steps += 1
        self.old_obs = new_obs

        return True

    def on_episode_end(self) -> None:
        self.num_episodes += 1

    @abstractmethod
    def on_training_start(self) -> None:
        pass

    @abstractmethod
    def on_training_end(self) -> None:
        pass
