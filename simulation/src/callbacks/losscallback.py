import os

import numpy as np
import pandas as pd
from cloudpickle import cloudpickle
from loguru import logger

from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback


class LossCallback(AbstractCallback):
    def __init__(self):
        super().__init__()
        self.metrics_tracker = None

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

    def on_update_end(self) -> None:
        if self.extra["agent_type"] in ["gpq_agent", "gpsarsa_agent"]:
            self.metrics_tracker.record_scalar("loss", self.agent_id, self.agent.latest_loss())
