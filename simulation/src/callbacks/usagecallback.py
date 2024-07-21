import pandas as pd

from zeus.monitor import ZeusMonitor
from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback
from util.memory_usage import get_gpu_memory_usage


class UsageCallback(AbstractCallback):

    def __init__(self):
        super().__init__()
        self.start_memory_usage = 0.0
        self.update_start_memory_usage = 0.0

        # We also want to look if time, memory and energy usage increases with each update.
        self.update_times = []
        self.update_memory = []
        self.update_energy = []

        self.energy_monitor = None

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
        self.start_memory_usage = 0.0
        self.update_start_memory_usage = 0.0

        # We also want to look if time, memory and energy usage increases with each update.
        self.update_times = []
        self.update_memory = []
        self.update_energy = []

        self.energy_monitor = ZeusMonitor()

    def _save_to_dataframe(self, execution_time, memory_usage, energy_usage):
        self.df.loc[self.df['agent_id'] == self.agent_id, "execution time (sec) " + self.mode] \
            = round(execution_time, 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "VRAM usage (GB) " + self.mode] \
            = round(memory_usage, 3)
        self.df.loc[self.df['agent_id'] == self.agent_id, "Energy usage (J) " + self.mode] \
            = round(energy_usage, 3)

    def on_training_start(self) -> None:
        self.start_memory_usage = get_gpu_memory_usage()[1]
        self.energy_monitor.begin_window("training")

    def on_step(self, action, reward, new_obs, done) -> bool:
        super().on_step(action, reward, new_obs, done)

        return True

    def on_episode_end(self) -> None:
        super().on_episode_end()

    def on_update_start(self) -> None:
        self.update_start_memory_usage = get_gpu_memory_usage()[1]
        self.energy_monitor.begin_window("updating")

    def on_update_end(self) -> None:
        update_memory_usage = get_gpu_memory_usage()[1] - self.start_memory_usage
        measurement = self.energy_monitor.end_window("updating")

        self.update_times.append(measurement.time)
        self.update_memory.append(update_memory_usage)
        self.update_energy.append(measurement.total_energy)

    def on_training_end(self) -> None:
        """
        You can plot here, but this is preferably done in the simulator class.
        :return:
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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the energy_monitor from the state to prevent serialization
        state['energy_monitor'] = None
        return state

    def __setstate__(self, state):
        # Restore the state and reinitialize energy_monitor
        self.__dict__.update(state)
        self.energy_monitor = ZeusMonitor()