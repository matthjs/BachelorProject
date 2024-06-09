import os
import signal
from typing import Optional

from loguru import logger
from stable_baselines3 import DQN, PPO

from agent.sbadapter import StableBaselinesAdapter
from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL
import cloudpickle
import atexit
import sys


def test_load_experiment():
    logger.debug("loading experiment...")

    (SimulatorRL("CartPole-v1", experiment_id="experiment_dummy_116")  # 75
     # .load_agent("gpq_agent_1", "gpq_agent")
     .load_agent("gpq_agent_1", "gpq_agent")
     # .load_agent("sb_dqn_1", "sb_dqn")
     # .load_agent("sb_ppo_1", "sb_ppo")
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .play("gpq_agent_1", 20))


class Backupper:
    def __init__(self, simulator: SimulatorRL = None):
        self.simulator = simulator
        atexit.register(self.backup_experiment)
        signal.signal(signal.SIGINT, self._exit_gracefully)

    """
    def save_all_sb_agents(self, backup_dir: str = "../backup/"):
        for agent_id, agent in self.simulator.agents.items():
            if agent.is_stable_baselines_wrapper():
                agent.stable_baselines_unwrapped().save(backup_dir + self.simulator.experiment_id + "_" + agent_id)

    def load_all_sb_agents(self, backup_dir: str = "../backup/"):
        for agent_id, agent in self.simulator.agents.items():
            if self.simulator.agents_info[agent_id]["agent_type"] == "sb_dqn":
                self.simulator.agents[agent_id] = StableBaselinesAdapter(
                    DQN.load(backup_dir + self.simulator.experiment_id + "_" + agent_id, env=self.simulator.env))
            elif self.simulator.agents_info[agent_id]["agent_type"] == "sb_ppo":
                self.simulator.agents[agent_id] = StableBaselinesAdapter(
                    PPO.load(backup_dir + self.simulator.experiment_id + "_" + agent_id, env=self.simulator.env))
    """

    def restore_experiment(self, experiment_id: str, new_experiment_id: Optional[str] = None,
                           directory: str = "../backup/") -> SimulatorRL:
        with open(directory + experiment_id + "_backup.pkl", "rb") as f:
            self.simulator: SimulatorRL = cloudpickle.load(f)

        if new_experiment_id is not None:
            self.simulator.experiment_id = new_experiment_id

        return self.simulator

    def _exit_gracefully(self, signum, frame):
        print("Exiting program with exit code -1")
        # Perform any cleanup actions here
        self.backup_experiment()  # Call your method or perform any necessary actions
        sys.exit(-1)

    def backup_experiment(self, directory: str = "../backup/") -> None:
        print("HELLO!")
        print(os.getcwd())
        with open(directory + self.simulator.experiment_id + "_backup.pkl", "wb") as f:
            cloudpickle.dump(self.simulator, f)


if __name__ == "__main__":
    b = Backupper()
    sim = b.restore_experiment("experiment_dummy_121")

    # sim = SimulatorRL("CartPole-v1", experiment_id="experiment_dummy_121")
    # b = Backupper(sim)

    (sim
     # .register_agent("gpq_agent_3", "gpq_agent")
     # .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_2", "gpsarsa_agent")
     # .register_agent("sb_dqn_1", "sb_dqn")
     #.register_agent("sb_dqn_2", "sb_dqn")
     # .register_agent("sb_ppo_1", "sb_ppo")
     .train_agents(agent_id_list=["gpq_agent_2"], num_episodes=30, concurrent=False,
                   callbacks=[RewardCallback(), UsageCallback()])
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .data_to_csv()
     .plot_any_plottable_data()
     .save_agents())

    # raise KeyboardInterrupt

    sim.play("gpq_agent_2", 10)
