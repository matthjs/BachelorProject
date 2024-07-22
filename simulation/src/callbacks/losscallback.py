import pandas as pd

from typing import Any
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry
from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback


class LossCallback(AbstractCallback):
    """
    Callback to track and record the loss of an agent during training.
    """

    def __init__(self) -> None:
        """
        Constructor for LossCallback.
        """
        super().__init__()
        self.metrics_tracker = None

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

    def on_update_end(self) -> None:
        """
        Callback at the end of an update to record the loss.

        This method records the loss of the agent if the agent type is "gpq_agent" or "gpsarsa_agent".
        """
        if self.extra and self.extra.get("agent_type") in ["gpq_agent", "gpsarsa_agent"]:
            self.metrics_tracker.record_scalar("loss", self.agent_id, self.agent.latest_loss())
