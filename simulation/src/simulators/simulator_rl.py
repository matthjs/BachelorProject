import pandas as pd
import gymnasium as gym

from agentfactory.agentfactory import AgentFactory
from metricstracker.metricstracker import MetricsTracker


class SimulatorRL:
    """
    Idea, have this class collect relevant information into a dataframe, which can
    be exported to a CSV.
    Allow for agent hyperparams to be configured using YAML file.
    In dataframe, at least put a comparison of mean reward test there with random
    policy (statistic value and P-value).
    Use Multithreading? May not scale as expected.

    """
    def __init__(self, env_str: str):
        self.df = pd.DataFrame()
        self.metrics_tracker = MetricsTracker()
        self.agents = {}
        self.agent_factory = AgentFactory()
        self.env_str = env_str
        self.env = gym.make(env_str)

        # We first want to record the performance of the random policy
        # so we can compare later
        self.agents["random"] = self.agent_factory.create_agent("random", env_str)

    def register_agent(self, agent_id: str, agent_type: str, cfg) -> 'SimulatorRL':
        agent = self.agent_factory.create_agent_configured(agent_type, cfg, self.env_str)
        self.agents[agent_id] = agent
        return self

    def register_env_str(self, env_str) -> 'SimulatorRL':
        self.env_str = env_str
        return self

    def train_stable_baselines_algorithm(self) -> 'SimulatorRL':
        return self

    def hyperopt_experiment(self):
        return self

    def train_agent(self, agent_id_list: list[str], concurrent=False):
        pass

    def evaluate_agent(self, agent_id_list: list[str], concurrent=False):
        pass

    def play(self, agent_id: str, num_episodes: int) -> None:
        agent = self.agents[agent_id]
        play_env = gym.make(self.env_str, render_mode='human')
        obs, info = play_env.reset()

        while True:
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, terminated, truncated, info = play_env.step(action)

            if terminated or truncated:
                num_episodes -= 1
                obs, info = play_env.reset()

            if num_episodes == 0:
                break

        play_env.close()



