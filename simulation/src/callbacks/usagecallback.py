import pandas as pd
from typing import Any
from zeus.monitor import ZeusMonitor
from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback
from util.memory_usage import get_gpu_memory_usage


class UsageCallback(AbstractCallback):
    """
    Callback to track and record the resource usage (time, memory, energy) during training.
    """

    def __init__(self) -> None:
        """
        Constructor for UsageCallback.
        """
        super().__init__()
        self.start_memory_usage: float = 0.0
        self.update_start_memory_usage: float = 0.0

        # Track time, memory, and energy usage during updates
        self.update_times: list = []
        self.update_memory: list = []
        self.update_energy: list = []

        self.energy_monitor: ZeusMonitor = None

    def init_callback(self,
                      experiment_id: str,
                      mode: str,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config: Any,
                      df: pd.DataFrame,
                      metrics_tracker_registry: Any,
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
        self.start_memory_usage = 0.0
        self.update_start_memory_usage = 0.0

        # Initialize tracking lists
        self.update_times = []
        self.update_memory = []
        self.update_energy = []

        self.energy_monitor = ZeusMonitor()

    def _save_to_dataframe(self, execution_time: float, memory_usage: float, energy_usage: float) -> None:
        """
        Save the resource usage metrics to the DataFrame.

        :param execution_time: Execution time in seconds.
        :param memory_usage: Memory usage in GB.
        :param energy_usage: Energy usage in joules.
        """
        self.df.loc[self.df['agent_id'] == self.agent_id, "execution time (sec) " + self.mode] = round(execution_time, 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "VRAM usage (GB) " + self.mode] = round(memory_usage, 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "Energy usage (J) " + self.mode] = round(energy_usage, 3)

    def on_training_start(self) -> None:
        """
        Callback at the start of training.
        """
        self.start_memory_usage = get_gpu_memory_usage()[1]
        self.energy_monitor.begin_window("training")

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
        return True

    def on_episode_end(self) -> None:
        """
        Callback at the end of an episode.
        """
        super().on_episode_end()

    def on_update_start(self) -> None:
        """
        Callback at the start of an update.
        """
        self.update_start_memory_usage = get_gpu_memory_usage()[1]
        self.energy_monitor.begin_window("updating")

    def on_update_end(self) -> None:
        """
        Callback at the end of an update.
        """
        update_memory_usage = get_gpu_memory_usage()[1] - self.start_memory_usage
        measurement = self.energy_monitor.end_window("updating")

        self.update_times.append(measurement.time)
        self.update_memory.append(update_memory_usage)
        self.update_energy.append(measurement.total_energy)

    def on_training_end(self) -> None:
        """
        Callback at the end of training.
        """
        memory_usage_diff = get_gpu_memory_usage()[1] - self.start_memory_usage
        measurement = self.energy_monitor.end_window("training")
        if self.logging:
            print(f"Training execution time {measurement.time:.3f} sec")
            print(f"Training VRAM usage {memory_usage_diff:.3f} GB")
            print(f"Training energy usage {measurement.total_energy:.3f} J")

        self._save_to_dataframe(measurement.time, memory_usage_diff, measurement.total_energy)

        if self.mode == "train":
            self.extra["update_times"] = self.update_times
            self.extra["update_energy"] = self.update_energy
            self.extra["update_memory"] = self.update_memory

    def __getstate__(self) -> dict:
        """
        Get the state for serialization.
        :return: The state dictionary.
        """
        state = self.__dict__.copy()
        # Remove the energy_monitor from the state to prevent serialization
        state['energy_monitor'] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Set the state after deserialization.
        :param state: The state dictionary.
        """
        self.__dict__.update(state)
        self.energy_monitor = ZeusMonitor()
