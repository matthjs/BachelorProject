import pandas as pd
import gymnasium as gym
from loguru import logger
from stable_baselines3.common.base_class import BaseAlgorithm

from agent.abstractagent import AbstractAgent
from agentfactory.agentfactory import AgentFactory
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry


class SimulatorRL:
    """
    Idea, have this class collect relevant information into a dataframe, which can
    be exported to a CSV.
    Allow for agent hyperparams to be configured using YAML file.
    In dataframe, at least put a comparison of mean reward test there with random
    policy (statistic value and P-value).
    Use Multithreading? May not scale as expected.

    I judged it to be simpler/less confusion to seperate custom agents and stable baselines algorithms
    as opposed to making wrapper classes

    """

    def __init__(self, env_str: str):
        self.df = pd.DataFrame()
        self.metrics_tracker_registry = MetricsTrackerRegistry()  # Singleton!
        self.metrics_tracker_registry.register_tracker("train")
        self.metrics_tracker_registry.register_tracker("eval")

        self._agents = {}
        self.agent_factory = AgentFactory()
        self.env_str = env_str
        self.env = gym.make(env_str)

        # We first want to record the performance of the random policy
        # so we can compare later
        self._agents["random"] = self.agent_factory.create_agent("random", env_str)

    def register_agent(self, agent_id: str, agent_type: str, cfg) -> 'SimulatorRL':
        agent = self.agent_factory.create_agent_configured(agent_type, cfg, self.env_str)
        self._agents[agent_id] = agent
        return self

    def data_to_csv(self):
        raise NotImplementedError()

    def plot_any_plottable_data(self):
        raise NotImplementedError()

    def register_load_agent(self, file: str) -> 'SimulatorRL':
        return self

    def register_env_str(self, env_str) -> 'SimulatorRL':
        self.env_str = env_str
        return self

    def hyperopt_experiment(self):
        return self

    def _test_with_random_policy(self):
        raise NotImplementedError()

    def train_agents(self, num_episodes: int, agent_id_list: list[str] = None, concurrent=False, logging=False) \
            -> 'SimulatorRL':
        if agent_id_list is None:
            agent_id_list = list(self._agents.keys())  # do all agents if agent_id_list not specified.

        self._stable_baselines_train("random", num_episodes)

        if not concurrent:
            for agent_id in agent_id_list:
                if agent_id not in self._agents:
                    raise ValueError("agent_id not in agents dictionary")

                if self._agents[agent_id].is_stable_baselines_wrapper():
                    self._stable_baselines_train(agent_id, num_episodes, logging)
                else:
                    self._agent_train_env_interaction_gym(agent_id, num_episodes, logging)
        else:
            raise NotImplementedError("OOPS I did not implement this yet my bad.")

        tracker = self.metrics_tracker_registry["train"]
        tracker.plot_return(title=self.env_str + "_" + agent_id_list.__str__())

        return self

    def evaluate_agents(self,
                        eval_env_str: str,
                        num_episodes: int,
                        agent_id_list: list[str] = None,
                        concurrent=False,
                        logging=False):
        raise NotImplementedError()

    def play(self, agent_id: str, num_episodes: int) -> None:
        agent = self._agents[agent_id]
        play_env = gym.make(self.env_str, render_mode='human')
        obs, info = play_env.reset()

        while True:
            old_obs = obs
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
        agent = self._agents[agent_id]
        tracker = self.metrics_tracker_registry.get_tracker("train")
        episode_reward = 0
        highest_average_return: float = -9999

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
                tracker.record_return(agent_id)
                episode_reward = 0
                obs, info = env.reset()

                current_avg_return = tracker.latest_average_return(agent_id)
                if current_avg_return > highest_average_return:
                    highest_average_return = current_avg_return
                    # Do something maybe

                if logging:
                    logger.debug(f"Episode reward / highest episode reward"
                                 f": {episode_reward} / {highest_average_return}")

            if num_episodes == 0:
                break

        env.close()

    def _stable_baselines_train(self, agent_id: str, num_episodes: int, logging=False) -> None:
        agent = self._agents[agent_id]
        if not agent.is_stable_baselines_wrapper():
            raise AttributeError("This function should only be run on wrapped StableBaselines agents")

        model = agent.stable_baselines_unwrapped()

        model.learn(total_timesteps=4000)  # TODO: Find a way to train SB agent for N episodes.
