import os

import pandas as pd
import gymnasium as gym
from loguru import logger
from scipy.stats import ranksums
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from agentfactory.agentfactory import AgentFactory
from callbacks.abstractcallback import AbstractCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.sbcallbackadapter import StableBaselinesCallbackAdapter
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry
from hydra import compose, initialize
import cloudpickle

from util.usageplotter import plot_complexities


class SimulatorRL:
    """
    Idea, have this class collect relevant information into a dataframe, which can
    be exported to a CSV.
    Allow for agent hyperparams to be configured using YAML file.
    In dataframe, at least put a comparison of mean reward test there with random
    policy (statistic value and P-value).
    Use Multithreading? May not scale as expected.

    """

    def __init__(self, env_str: str, experiment_id="simulation", verbose: int = 2):
        self.df = pd.DataFrame()
        self.metrics_tracker_registry = MetricsTrackerRegistry()  # Singleton!
        self.metrics_tracker_registry.register_tracker("train")
        self.metrics_tracker_registry.get_tracker("train").register_metric("return")
        self.metrics_tracker_registry.register_tracker("eval")
        self.metrics_tracker_registry.get_tracker("eval").register_metric("return")

        self.experiment_id = experiment_id
        self.verbose = verbose

        self.agents = {}
        self.agents_configs = {}
        self.agents_info = {}

        self.agent_factory = AgentFactory()
        self.env_str = env_str
        self.env = gym.make(env_str)

        # We first want to record the performance of the random policy
        # so we can compare later
        self.agents["random"] = self.agent_factory.create_agent("random", env_str)
        self.agents_configs["random"] = None
        self._add_agent_to_df("random", "random")
        self.agents_info["random"] = {}

    def _config_obj(self, agent_type: str, agent_id: str, env_str: str, config_path: str = "../../../configs"):
        with initialize(config_path=config_path + "/" + agent_type, version_base="1.2"):
            cfg = compose(config_name="config_" + agent_id + "_" + env_str)

        self.agents_configs[agent_id] = cfg
        return cfg

    def _add_agent_to_df(self, agent_id: str, agent_type: str, hyperparams=None) -> None:
        if hyperparams is None:
            hyperparams = {}
        new_row_values = {"agent_id": agent_id, "agent_type": agent_type, "hyperparams": hyperparams}
        new_row_df = pd.DataFrame([new_row_values])
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

    def _signtest_train_with_random_policy(self, agent_id: str):
        tracker = self.metrics_tracker_registry.get_tracker("train")
        means_random, _ = tracker.metric_history("return").get("random")
        means, _ = tracker.metric_history("return").get(agent_id)
        statistic, p_value = ranksums(means, means_random, alternative='greater')

        if self.verbose > 0:
            logger.info(f"{agent_id} p-value (cuml): {p_value}")

        self.df.loc[self.df['agent_id'] == agent_id, "p_val_cuml_return_train_>"] = round(p_value, 3)
        self.df.loc[self.df['agent_id'] == agent_id, "W-statistic_cuml"] = round(statistic, 3)

    def _signtest_eval_with_random_policy(self, agent_id: str):
        tracker = self.metrics_tracker_registry.get_tracker("eval")
        return_history_random = tracker.value_history("return").get("random")
        return_history = tracker.value_history("return").get(agent_id)
        statistic, p_value = ranksums(return_history, return_history_random, alternative='greater')

        if self.verbose > 0:
            logger.info(f"{agent_id} p-value (returns eval): {p_value}")

        self.df.loc[self.df['agent_id'] == agent_id, "p_val_return_eval_>"] = round(p_value, 3)
        self.df.loc[self.df['agent_id'] == agent_id, "W-statistic_return_eval"] = round(statistic, 3)

    def register_agent(self, agent_id: str, agent_type: str) -> 'SimulatorRL':
        cfg = self._config_obj(agent_type, agent_id, self.env_str)

        agent, hyperparams = self.agent_factory.create_agent_configured(agent_type, self.env_str, cfg)
        self.agents[agent_id] = agent

        self._add_agent_to_df(agent_id, agent_type, hyperparams)
        self.agents_info[agent_id] = {}

        return self

    def load_agent(self, agent_id: str, agent_type: str, data_dir="../data/saved_agents/") -> 'SimulatorRL':
        self._config_obj(agent_type, agent_id, self.env_str)  # side effect agents_config[agent_id] = cfg

        with open(data_dir + self.experiment_id + "/" + agent_id + ".pkl", "rb") as f:
            self.agents[agent_id] = cloudpickle.load(f)

        self._add_agent_to_df(agent_id, agent_type)
        self.agents_info[agent_id] = {}

        return self

    def load_data_from_csv(self, data_name: str = "simulation", data_path: str = "../data/experiments") \
            -> 'SimulatorRL':
        self.df = pd.read_csv(data_path + data_name + ".csv")
        return self

    def data_to_csv(self, data_path: str = "../data/experiments") -> 'SimulatorRL':
        # print(os.getcwd())
        self.df.to_csv(data_path + "/" + self.experiment_id + ".csv", index=False)
        return self

    def plot_any_plottable_data(self, plot_dir: str = "../plots/") -> 'SimulatorRL':
        if self.verbose > 1:
            logger.info("Plotting plottable data")

        tracker = self.metrics_tracker_registry.get_tracker("train")
        tracker.plot_metric(metric_name="return",
                            plot_path=plot_dir + self.env_str + self.experiment_id,
                            title=self.env_str + "_" + list(self.agents.keys()).__str__())

        for agent_id, info in self.agents_info.items():
            for info_attr, value in info.items():
                if info_attr == "update_times":
                    plot_complexities(value,
                                      f"Time Usage {agent_id}",
                                      "Update Index",
                                      "Time (seconds)",
                                      plot_dir=plot_dir + "/" + self.experiment_id)
                elif info_attr == "update_energy":
                    plot_complexities(value,
                                      f"Energy Usage {agent_id}",
                                      "Update Index",
                                      "Energy (Joule)",
                                      plot_dir=plot_dir + "/" + self.experiment_id)
                elif info_attr == "update_memory":
                    plot_complexities(value,
                                      f"VRAM usage {agent_id}",
                                      "Update Index",
                                      "Memory Usage (GB)",
                                      plot_dir=plot_dir + "/" + self.experiment_id)

        return self

    def register_load_agent(self, file: str) -> 'SimulatorRL':
        raise NotImplementedError()

    def register_env_str(self, env_str) -> 'SimulatorRL':
        self.env_str = env_str
        return self

    def hyperopt_experiment(self):
        return self

    def save_agents(self, agent_id_list: list[str] = None, save_dir="../data/saved_agents/") -> 'SimulatorRL':
        save_dir = save_dir + self.experiment_id + "/"

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        if self.verbose > 1:
            logger.info(f"Saving agents to {save_dir}...")

        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        # I am being really lazy here. Try and see if cloudpickle suffices.
        # Even though for StableBaselines agents there is the .save() .load methods.
        for agent_id in agent_id_list:
            with open(save_dir + agent_id + ".pkl", "wb") as f:
                cloudpickle.dump(self.agents[agent_id], f)

        if self.verbose > 1:
            logger.info("Done!")

        return self

    def train_agents(self,
                     num_episodes: int,
                     agent_id_list: list[str] = None,
                     callbacks: list[AbstractCallback] = None,
                     concurrent=False) \
            -> 'SimulatorRL':
        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if self.verbose > 1:
            logger.info(f"Training agents -> {agent_id_list}")

        if callbacks is None:
            callbacks = [RewardCallback()]

        if not concurrent:
            for agent_id in agent_id_list:
                if self.verbose > 1:
                    logger.info(f"Training agent -> {agent_id}")

                if agent_id not in self.agents:
                    raise ValueError("agent_id not in agents dictionary")

                # Add info to dataframe
                self.df.loc[self.df['agent_id'] == agent_id, "num_episodes (train)"] = num_episodes

                if self.agents[agent_id].is_stable_baselines_wrapper():
                    self._stable_baselines_train(agent_id, num_episodes, callbacks)
                else:
                    self._agent_env_interaction_gym("train", agent_id, num_episodes, callbacks)
        else:
            raise NotImplementedError("OOPS I did not implement this yet my bad.")

        for agent_id in agent_id_list:
            self._signtest_train_with_random_policy(agent_id)

        return self

    def evaluate_agents(self,
                        num_episodes: int,
                        agent_id_list: list[str] = None,
                        callbacks: list[AbstractCallback] = None,
                        concurrent=False) -> 'SimulatorRL':
        if callbacks is None:
            callbacks = [RewardCallback()]

        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if self.verbose > 1:
            logger.info(f"Evaluating agents -> {agent_id_list}")

        if not concurrent:
            for agent_id in agent_id_list:
                if self.verbose > 1:
                    logger.info(f"Evaluating -> {agent_id}")

                # Add info to dataframe
                self.df.loc[self.df['agent_id'] == agent_id, "num_episodes (eval)"] = num_episodes

                if agent_id not in self.agents:
                    raise ValueError("agent_id not in agents dictionary")

                self._agent_env_interaction_gym("eval", agent_id, num_episodes, callbacks)
        else:
            raise NotImplementedError("OOPS I did not implement this yet my bad.")

        for agent_id in agent_id_list:
            self._signtest_eval_with_random_policy(agent_id)

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

    def _agent_env_interaction_gym(self,
                                   mode: str,
                                   agent_id: str,
                                   num_episodes: int,
                                   callbacks: list[AbstractCallback]) -> None:
        """
        So basically, this is a train loop for the agent. Note that even though the update method is
        run at every time step, this does *not* mean the agent performs the update rule at every time step (batching).
        :param env:
        :param agent_id:
        :param num_episodes:
        :param logging:
        :return:
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode {mode}.")

        env = self.env
        agent = self.agents[agent_id]

        for callback in callbacks:
            callback.init_callback(
                self.experiment_id,
                mode,
                agent,
                agent_id,
                self.agents_configs[agent_id],
                self.df,
                self.metrics_tracker_registry,
                self.verbose > 0,
                self.agents_info[agent_id]
            )

        obs, info = env.reset()

        if mode == "train" and agent.is_stable_baselines_wrapper():
            raise AttributeError("This function in train mode should only be run on True custom agents")

        for callback in callbacks:
            callback.on_training_start()

        while True:
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            agent.record_env_info(info, terminated or truncated)
            agent.add_trajectory((old_obs, action, reward, obs))

            if mode == "train":
                if agent.updatable():
                    for callback in callbacks:
                        callback.on_update_start()
                    agent.update()
                    for callback in callbacks:
                        callback.on_update_end()
                else:
                    agent.update()

            for callback in callbacks:
                callback.on_step(action, reward, obs, terminated or truncated)

            if terminated or truncated:
                num_episodes -= 1
                for callback in callbacks:
                    callback.on_episode_end()
                obs, info = env.reset()

            if num_episodes == 0:
                break

        env.close()
        for callback in callbacks:
            callback.on_training_end()

    def _stable_baselines_train(self,
                                agent_id: str,
                                num_episodes: int,
                                callbacks: list[AbstractCallback]) -> None:
        agent = self.agents[agent_id]
        if not agent.is_stable_baselines_wrapper():
            raise AttributeError("This function should only be run on wrapped StableBaselines agents")

        sb_callbacks = [StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=0)]

        for callback in callbacks:
            callback.init_callback(
                self.experiment_id,
                "train",
                agent,
                agent_id,
                self.agents_configs[agent_id],
                self.df,
                self.metrics_tracker_registry,
                self.verbose > 0,
                self.agents_info[agent_id]
            )
            # Adapter pattern, ensure custom callbacks are compatible with
            # stable baselines callbacks.
            sb_callbacks.append(StableBaselinesCallbackAdapter(callback))

        model = agent.stable_baselines_unwrapped()

        # Problem: DQN's epsilon-greedy schedule depends on total_timesteps
        # parameter, which complicates the use of StopTrainingOnMaxEpisodes.
        # For instance, if total_timesteps is set arbitrarily large then
        # DQN will only execute random actions.
        model.learn(total_timesteps=int(5e4), callback=sb_callbacks)
        # model.learn(total_timesteps=int(216942042), callback=sb_callbacks)
