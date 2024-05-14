from datetime import time

import numpy as np
import pandas as pd
import gymnasium as gym
from loguru import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from agent.abstractagent import AbstractAgent
from agentfactory.agentfactory import AgentFactory
from callbacks.sbcallbackadapter import StableBaselinesCallbackAdapter
from metricstracker.metricstracker2 import MetricsTracker2
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry
from hydra import compose, initialize

class SimulatorRL:
    """
    Idea, have this class collect relevant information into a dataframe, which can
    be exported to a CSV.
    Allow for agent hyperparams to be configured using YAML file.
    In dataframe, at least put a comparison of mean reward test there with random
    policy (statistic value and P-value).
    Use Multithreading? May not scale as expected.

    I judged it to be simpler/less confusion to separate custom agents and stable baselines algorithms
    as opposed to making wrapper classes

    """

    def __init__(self, env_str: str):
        self.df = pd.DataFrame()
        self.metrics_tracker_registry = MetricsTrackerRegistry()  # Singleton!
        self.metrics_tracker_registry.register_tracker("train")
        self.metrics_tracker_registry.get_tracker("train").register_metric("return")
        self.metrics_tracker_registry.register_tracker("eval")
        self.metrics_tracker_registry.get_tracker("eval").register_metric("return")

        self.agents = {}
        self.agents_configs = {}

        self.agent_factory = AgentFactory()
        self.env_str = env_str
        self.env = gym.make(env_str)

        # We first want to record the performance of the random policy
        # so we can compare later
        self.agents["random"] = self.agent_factory.create_agent("random", env_str)

    def _config_obj(self, agent_type: str, agent_id: str, env_str: str, config_path: str = "../../../configs"):
        with initialize(config_path=config_path + "/" + agent_type):
            cfg = compose(config_name="config_" + agent_id + "_" + env_str)

        self.agents_configs[agent_id] = cfg
        return cfg

    def register_agent(self, agent_id: str, agent_type: str) -> 'SimulatorRL':
        cfg = self._config_obj(agent_type, agent_id, self.env_str)

        agent = self.agent_factory.create_agent_configured(agent_type, self.env_str, cfg)
        self.agents[agent_id] = agent
        return self

    def data_to_csv(self) -> 'SimulatorRL':
        return self

    def plot_any_plottable_data(self) -> 'SimulatorRL':
        tracker = self.metrics_tracker_registry.get_tracker("train")
        tracker.plot_metric(metric_name="return", title=self.env_str + "_" + list(self.agents.keys()).__str__())
        return self

    def register_load_agent(self, file: str) -> 'SimulatorRL':
        raise NotImplementedError()

    def register_env_str(self, env_str) -> 'SimulatorRL':
        self.env_str = env_str
        return self

    def hyperopt_experiment(self):
        return self

    def _test_with_random_policy(self):
        raise NotImplementedError()

    def train_agents(self, num_episodes: int, agent_id_list: list[str] = None, concurrent=False, logging=False) \
            -> 'SimulatorRL':
        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if not concurrent:
            for agent_id in agent_id_list:
                if agent_id not in self.agents:
                    raise ValueError("agent_id not in agents dictionary")

                if self.agents[agent_id].is_stable_baselines_wrapper():
                    self._stable_baselines_train(agent_id, num_episodes, logging)
                else:
                    self._agent_train_env_interaction_gym(agent_id, num_episodes, logging)
        else:
            raise NotImplementedError("OOPS I did not implement this yet my bad.")

        return self

    def evaluate_episode_reward(self, num_episodes: int, agent_id: str, eval_env: gym.Env) -> tuple[float, float]:
        """
        Return mean and std of the return (undiscounted sum of rewards) G = R_1 + R_2 + ... R_M over n episodes.
        :param eval_env:
        :param num_episodes:
        :param agent_id:
        :return:
        """
        tracker = self.metrics_tracker_registry.get_tracker("eval")
        agent = self.agents[agent_id]
        obs, info = eval_env.reset()

        episode_reward = 0

        while True:
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                num_episodes -= 1
                tracker.record_scalar("return", agent_id, episode_reward)
                episode_reward = 0
                obs, info = eval_env.reset()

            if num_episodes == 0:
                break

        eval_env.close()
        return tracker.latest_mean_variance("return", agent_id)


    def evaluate_agents(self,
                        eval_env_str: str,
                        num_episodes: int,
                        agent_id_list: list[str] = None,
                        concurrent=False,
                        logging=True,
                        plotting=False) -> 'SimulatorRL':
        eval_env = gym.make(eval_env_str)

        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if not concurrent:
            for agent_id in agent_id_list:
                if agent_id not in self.agents:
                    raise ValueError("agent_id not in agents dictionary")

                mean, var = self.evaluate_episode_reward(
                    num_episodes,
                    agent_id,
                    eval_env
                )
                if logging:
                    print(f"Avg return ({num_episodes} ep) - {agent_id} - {mean:.3f} +- {np.sqrt(var):.3f}")
        else:
            raise NotImplementedError("OOPS I did not implement this yet my bad.")

        return self

    def play(self, agent_id: str, num_episodes: int) -> None:
        agent = self.agents[agent_id]
        play_env = gym.make(self.env_str, render_mode='human')
        obs, info = play_env.reset()

        while True:
            action = agent.policy(obs)  # Will run .predict() if this is actually StableBaselines algorithm.
            obs, reward, terminated, truncated, info = play_env.step(action)

            if terminated or truncated:
                num_episodes -= 1
                obs, info = play_env.reset()

            if num_episodes == 0:
                break

        play_env.close()

    def _agent_train_env_interaction_gym(self, agent_id: str, num_episodes: int, logging=False) -> None:
        """
        So basically, this is a train loop for the agent. Note that even though the update method is
        run at every time step, this does *not* mean the agent performs the update rule at every time step (batching).
        :param env:
        :param agent_id:
        :param num_episodes:
        :param logging:
        :return:
        """
        env = self.env
        agent = self.agents[agent_id]
        tracker = self.metrics_tracker_registry.get_tracker("train")
        episode_reward = 0
        highest_average_return: float = -9999

        obs, info = env.reset()

        if agent.is_stable_baselines_wrapper():
            raise AttributeError("This function should only be run on True custom agents")

        while True:
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            agent.record_env_info(info, terminated or truncated)
            agent.add_trajectory((old_obs, action, reward, obs))

            episode_reward += reward

            agent.update()

            if terminated or truncated:
                num_episodes -= 1
                tracker.record_scalar("return", agent_id, episode_reward)
                obs, info = env.reset()

                current_avg_return, _ = tracker.latest_mean_variance("return", agent_id)
                if current_avg_return > highest_average_return:
                    highest_average_return = current_avg_return
                    # Do something maybe

                if logging:
                    logger.debug(f"Episode reward / highest episode reward"
                                 f": {episode_reward} / {highest_average_return}")

                episode_reward = 0

            if num_episodes == 0:
                break

        env.close()

    def _stable_baselines_train(self, agent_id: str, num_episodes: int, logging=False) -> None:
        # TODO: Record info such that I can actually compare properly with custom agents.
        agent = self.agents[agent_id]
        if not agent.is_stable_baselines_wrapper():
            raise AttributeError("This function should only be run on wrapped StableBaselines agents")

        model = agent.stable_baselines_unwrapped()

        model.learn(total_timesteps=9999999999999, callback=[StableBaselinesCallbackAdapter(None), StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=0)])