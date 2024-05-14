import time

import pandas as pd
import resource
from agent.abstractagent import AbstractAgent
from callbacks.abstractcallback import AbstractCallback


# Function to measure memory usage
def get_memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class UsageCallback(AbstractCallback):
    """
    Warning: memory usage functionality probably not that reliable.
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.memory_before = None

    def init_callback(self,
                      mode: str,
                      agent: AbstractAgent,
                      agent_id: str,
                      agent_config,
                      df: pd.DataFrame,
                      metrics_tracker_registry,
                      logging=False,
                      extra=None):
        super().init_callback(mode, agent, agent_id, agent_config, df, metrics_tracker_registry, logging, extra)

    def on_training_start(self) -> None:
        self.start_time = time.time()
        self.memory_before = get_memory_usage()

    def on_step(self, action, reward, new_obs, done) -> bool:
        super().on_step(action, reward, new_obs, done)

        return True

    def on_episode_end(self) -> None:
        super().on_episode_end()

    def on_training_end(self) -> None:
        """
        You can plot here, but this is preferably done in the simulator class.
        :return:
        """
        end_time = time.time()
        execution_time = end_time - self.start_time
        memory_usage_diff = get_memory_usage() - self.memory_before
        if self.logging:
            print(f"Execution time {execution_time:.3f} sec")
            print(f"Mem usage change {memory_usage_diff} KB")
