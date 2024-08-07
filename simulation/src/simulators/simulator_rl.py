import os
import pandas as pd
import cloudpickle

from typing import Optional
from loguru import logger
from scipy.stats import ranksums
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
from agent.sbadapter import StableBaselinesAdapter
from agentfactory.agentfactory import AgentFactory
from callbacks.abstractcallback import AbstractCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.sbcallbackadapter import StableBaselinesCallbackAdapter
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry
from hydra import compose, initialize
from util.make_vec_normalized_env import make_vec_normalized_env, make_vec_env


def is_zip_file(filename: str) -> bool:
    return filename.lower().endswith('.zip')


class SimulatorRL:
    """
    Class for managing RL experiments.
    I am aware that a class this big is bad design, but it is what it is.
    """

    def __init__(self, env_str: str, experiment_id="simulation", verbose: int = 2):
        """
        Initialize the SimulatorRL instance.

        :param env_str: The environment string identifier for the Gym environment.
        :param experiment_id: Identifier for the experiment. Default is "simulation".
        :param verbose: Verbosity level. Default is 2.
        """
        self.df = pd.DataFrame()

        self.metrics_tracker_registry = MetricsTrackerRegistry()
        self.metrics_tracker_registry.register_tracker("train")
        self.metrics_tracker_registry.get_tracker("train").register_metric("return")
        self.metrics_tracker_registry.get_tracker("train").register_metric("loss")
        self.metrics_tracker_registry.register_tracker("eval")
        self.metrics_tracker_registry.get_tracker("eval").register_metric("return")

        self.experiment_id = experiment_id
        self.verbose = verbose

        self.agents = {}
        self.agents_configs = {}
        self.agents_info = {}

        self.agent_factory = AgentFactory()
        self.env_str = env_str
        self.env = None
        self.eval_env = None
        self._initialize_env(env_str)

        # We first want to record the performance of the random policy
        # so we can compare later
        self.init_random_agent()
        self.callback_ref = None

    def init_random_agent(self):
        """
        Initialize the random agent and add it to the dataframe and agents_info dictionary.
        """
        self.agents["random"] = self.agent_factory.create_random_agent(self.env_str)
        self.agents_configs["random"] = None
        self._add_agent_to_df("random", "random")
        self.agents_info["random"] = {}
        self.agents_info["random"]["agent_type"] = "random"

    @staticmethod
    def load(experiment_id: str,
             new_experiment_id: Optional[str] = None,
             load_dir: str = "../data/simulationbackup") -> 'SimulatorRL':
        """
        Load a SimulatorRL instance from a backup file.
        (1) load simulator.
        (2) load agents that were used in the experiment.

        :param experiment_id: Identifier for the experiment to be loaded.
        :param new_experiment_id: Optional new identifier for the experiment.
        :param load_dir: Directory where the backup file is located. Default is "../data/simulationbackup".
        :return: Loaded SimulatorRL instance.
        """
        logger.info(f"loading simulator with experiment id: {experiment_id}")
        with open(load_dir + "/" + experiment_id + "_backup.pkl", "rb") as f:
            simulator: SimulatorRL = cloudpickle.load(f)

        simulator.agents = {}

        agent_id_list = simulator.agent_id_list()
        for agent_id in agent_id_list:
            if agent_id != "random":
                simulator.load_agent(agent_id, simulator.agents_info[agent_id]["agent_type"])

        simulator.init_random_agent()

        logger.info("done!")

        simulator.env = VecNormalize.load(load_dir + "/env" + simulator.experiment_id + ".pkl",
                                          make_vec_env(simulator.env_str))
        simulator.eval_env = VecNormalize.load(load_dir + "/eval_env" + simulator.experiment_id + ".pkl",
                                               make_vec_env(simulator.env_str))

        if new_experiment_id is not None:
            logger.info("replacing experiment id with: ", new_experiment_id)
            simulator.experiment_id = new_experiment_id

        return simulator

    def save(self, save_dir: str = "../data/simulationbackup") -> None:
        """
        Save the SimulatorRL instance to a backup file.
        NOTE: the self.agents field is deleted, those agents are pickled themselves. The load static method
        takes this into account.

        :param save_dir: Directory where the backup file will be saved. Default is "../data/simulationbackup".
        """
        self.save_agents()
        del self.agents
        if self.verbose > 1:
            logger.info("Saving simulator...")
        with open(save_dir + "/" + self.experiment_id + "_backup.pkl", "wb") as f:
            cloudpickle.dump(self, f)
        if self.verbose > 1:
            logger.info("Done!")
        self.env.save(save_dir + "/env" + self.experiment_id + ".pkl")
        self.eval_env.save(save_dir + "/eval_env" + self.experiment_id + ".pkl")

    def load_agent(self, agent_id: str, agent_type: str, data_dir="../data/saved_agents/") -> 'SimulatorRL':
        """
        Load an agent from disk.

        :param agent_id: Identifier of the agent to load.
        :param agent_type: Type of the agent.
        :param data_dir: Directory path where the agent files are stored, defaults to "../data/saved_agents/".

        :return: Instance of SimulatorRL with the loaded agent.
        """
        self._config_obj(agent_type, agent_id, self.env_str)  # side effect agents_config[agent_id] = cfg

        path = data_dir + self.experiment_id + "/" + agent_id

        stupid_flag = True

        if os.path.exists(path + "_sb_dqn.zip"):
            stupid_flag = False
            self.agents[agent_id] = StableBaselinesAdapter(DQN.load(path + "_sb_dqn.zip", env=self.env))
        elif os.path.exists(path + "_sb_ppo.zip"):
            stupid_flag = False
            self.agents[agent_id] = StableBaselinesAdapter(PPO.load(path + "_sb_ppo.zip", env=self.env))

        if stupid_flag:
            with open(path + ".pkl", "rb") as f:
                self.agents[agent_id] = cloudpickle.load(f)

        self._add_agent_to_df(agent_id, agent_type)
        self.agents_info[agent_id] = {}
        self.agents_info[agent_id]["agent_type"] = agent_type

        return self

    def save_agents(self, agent_id_list: list[str] = None, save_dir="../data/saved_agents/") -> 'SimulatorRL':
        """
        Save agents to disk.

        :param agent_id_list: List of agent identifiers to save. If None, all agents are saved.
        :param save_dir: Directory path to save the agents, defaults to "../data/saved_agents/".

        :return: Instance of SimulatorRL with agents saved.
        """
        save_dir = save_dir + self.experiment_id + "/"

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        if self.verbose > 1:
            logger.info(f"Saving agents to {save_dir}...")

        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        # Save agents to stable storage.
        for agent_id in agent_id_list:
            if self.agents[agent_id].is_stable_baselines_wrapper():
                self.agents[agent_id].stable_baselines_unwrapped().save(save_dir + agent_id + "_" + \
                                                                        self.agents_info[agent_id]["agent_type"])
            else:
                with open(save_dir + agent_id + ".pkl", "wb") as f:
                    cloudpickle.dump(self.agents[agent_id], f)

        if self.verbose > 1:
            logger.info("Done!")

        return self

    def register_agent(self, agent_id: str, agent_type: str) -> 'SimulatorRL':
        """
        Register an agent with the simulator.

        :param agent_id: Identifier of the agent.
        :param agent_type: Type of the agent.
        :return: Instance of SimulatorRL with the registered agent.
        """
        cfg = self._config_obj(agent_type, agent_id, self.env_str)

        agent, hyperparams = self.agent_factory.create_agent_configured(agent_type, self.env, cfg)
        self.agents[agent_id] = agent

        self._add_agent_to_df(agent_id, agent_type, hyperparams)
        self.agents_info[agent_id] = {}
        self.agents_info[agent_id]["agent_type"] = agent_type

        return self

    def plot_any_plottable_data(self,
                                agent_id_list: Optional[list[str]] = None,
                                color_list: Optional[dict] = None,
                                plot_dir: str = "../plots/") -> 'SimulatorRL':
        """
        Plot any plottable data and save the plots to files.

        :param agent_id_list: List of agent identifiers to plot. If None, plot all agents.
        :param color_list: Dictionary of agent identifiers to colors.
        :param plot_dir: Directory where the plots will be saved. Default is "../plots/".
        :return: The SimulatorRL instance.
        """
        if self.verbose > 1:
            logger.info("Plotting plottable data")

        tracker = self.metrics_tracker_registry.get_tracker("train")
        tracker.plot_metric(metric_name="return",
                            plot_path=plot_dir + self.env_str + self.experiment_id,
                            title="graph " + self.env_str,
                            id_list=agent_id_list,
                            color_list=color_list)
        tracker.plot_metric(metric_name="loss",
                            x_axis_label="updates",
                            plot_path=plot_dir + self.env_str + self.experiment_id + "_loss",
                            title="graph " + self.env_str,
                            id_list=agent_id_list,
                            color_list=color_list)

        return self

    def train_agents(self,
                     num_episodes: int,
                     agent_id_list: list[str] = None,
                     callbacks: list[AbstractCallback] = None,
                     concurrent=False,
                     reuse_callbacks=False) \
            -> 'SimulatorRL':
        """
        Train specified or all agents for a given number of episodes.

        :param num_episodes: Number of episodes for training.
        :param agent_id_list: List of agent identifiers to be trained. If None, all agents will be trained.
        :param callbacks: List of callback functions to be used during training.
        :param concurrent: Flag indicating if training should be done concurrently [NOT IMPLEMENTED]. Default is False.
        :param reuse_callbacks: Flag indicating if callbacks should be reused. Default is False.
        :return: The SimulatorRL instance after training.
        """
        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if self.verbose > 1:
            logger.info(f"Training agents -> {agent_id_list}")

        if callbacks is None:
            callbacks = [RewardCallback()]
        if reuse_callbacks:
            callbacks = self.callback_ref

        self.callback_ref = callbacks

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

        if len(callbacks) != 0:
            for agent_id in agent_id_list:
                self._signtest_train_with_random_policy(agent_id)

        return self

    def evaluate_agents(self,
                        num_episodes: int,
                        agent_id_list: list[str] = None,
                        callbacks: list[AbstractCallback] = None,
                        concurrent=False) -> 'SimulatorRL':
        """
        Evaluate specified or all agents for a given number of episodes.

        :param num_episodes: Number of episodes for evaluation.
        :param agent_id_list: List of agent identifiers to be evaluated. If None, all agents will be evaluated.
        :param callbacks: List of callback functions to be used during evaluation.
        :param concurrent: Flag indicating if evaluation should be done concurrently. Default is False.
        :return: The SimulatorRL instance after evaluation.
        """
        self._eval_start()
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

        if len(callbacks) != 0:
            for agent_id in agent_id_list:
                self._signtest_eval_with_random_policy(agent_id)

        self._eval_end()
        return self

    def record(self, agent_id: str, num_timeseps: int, record_dir: str = "../videos/") -> None:
        """
        Record a video of the agent's performance.

        :param agent_id: Identifier of the agent to be recorded.
        :param num_timesteps: Number of timesteps to record.
        :param record_dir: Directory where the video will be saved. Default is "../videos/".
        """
        self._eval_start()
        # self.env.render_mode = "rgb_array"
        agent = self.agents[agent_id]
        obs = self.env.reset()[0]
        play_env = VecVideoRecorder(venv=self.env,
                                    video_folder=record_dir,
                                    record_video_trigger=lambda x: x == 0,
                                    video_length=num_timeseps,
                                    name_prefix=self.experiment_id + "_" + agent_id)

        episode_reward = 0

        for _ in range(num_timeseps - 1):
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, done, info = play_env.step([action])
            obs = obs[0]  # (1, state_dim) -> (state_dim)
            reward = reward[0]  # (1,) -> float
            episode_reward += reward

            if done:
                logger.info(f"Episode reward {episode_reward}")
                episode_reward = 0
                obs = play_env.reset()[0]

        self._eval_end()

    def _agent_env_interaction_gym(self,
                                   mode: str,
                                   agent_id: str,
                                   num_episodes: int,
                                   callbacks: list[AbstractCallback]) -> None:
        """
        Train or evaluate an agent in the environment.

        :param mode: The mode of operation, either 'train' or 'eval'.
        :param agent_id: Identifier of the agent.
        :param num_episodes: Number of episodes for training or evaluation.
        :param callbacks: List of callback functions to be used during training or evaluation.
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

        obs = env.reset()[0]

        if mode == "train" and agent.is_stable_baselines_wrapper():
            logger.warning("This function in train mode does not train StableBaselines3 agents")

        for callback in callbacks:
            callback.on_training_start()

        ep = num_episodes

        while True:
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, done, info = env.step([action])
            obs = obs[0]  # (1, state_dim) -> (state_dim)
            reward = reward[0]  # (1,) -> float

            agent.add_trajectory((old_obs, action, reward, obs))

            if mode == "train" and not agent.is_stable_baselines_wrapper():
                if agent.updatable():
                    for callback in callbacks:
                        callback.on_update_start()
                    agent.update()
                    for callback in callbacks:
                        callback.on_update_end()
                else:
                    agent.update()

            for callback in callbacks:
                callback.on_step(action, reward, obs, done)

            if done:
                num_episodes -= 1
                for callback in callbacks:
                    callback.on_episode_end()

                obs = env.reset()[0]
                # NOTE: Logging frequency hardcoded.
                if (ep - num_episodes) % 30 == 0 and self.verbose > 0:
                    tracker = self.metrics_tracker_registry.get_tracker("train")
                    tracker.plot_metric(metric_name="return",
                                        plot_path="../plots/" + self.env_str + self.experiment_id,
                                        title=f"{self.env_str}_{', '.join(self.agents.keys())}")
                    logger.debug(f"Episode: {ep - num_episodes}")

            if num_episodes == 0:
                break

        env.close()
        for callback in callbacks:
            callback.on_training_end()

    def _stable_baselines_train(self,
                                agent_id: str,
                                num_episodes: int,
                                callbacks: list[AbstractCallback]) -> None:
        """
        Train the agent using the Stable Baselines library.

        :param agent_id: Identifier of the agent.
        :param num_episodes: Number of episodes for training.
        :param callbacks: List of callback functions to be used during training.
        """
        agent = self.agents[agent_id]
        if not agent.is_stable_baselines_wrapper():
            raise AttributeError("This function should only be run on wrapped StableBaselines agents")

        max_episode_callback = StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=0)
        sb_callbacks = [max_episode_callback]

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

        if self.env_str == "CartPole-v1":
            time_steps = 5e4
        elif self.env_str == "LunarLander-v2":
            time_steps = 1e5
        else:
            logger.warning(
                "Warning: Number of timesteps for DQN are currently hardcoded for CartPole and Lunar Lander. "
                "Defaults to arbitrarily large timesteps as there always is a maximum number of episodes. "
                "However, DQN's epsilon-greedy schedule depends on the total_timesteps parameter, complicating "
                "the use of StopTrainingOnMaxEpisodes. If total_timesteps is set arbitrarily large, DQN will only "
                "execute random actions. To add support for DQN in other environments,"
                "re-adjust the code to make the number of timesteps part of the configuration."
            )
            time_steps = 4242424242424

        model.learn(total_timesteps=int(time_steps), callback=sb_callbacks)

        self._eval_start()
        # this does not train StableBaselines3 agents.
        self._agent_env_interaction_gym("train",
                                        agent_id,
                                        num_episodes - max_episode_callback.n_episodes,
                                        callbacks)
        self._eval_end()

    def _eval_start(self):
        self.env.training = False

    def _eval_end(self):
        self.env.training = True

    def _initialize_env(self, env_str: str, config_path: str = "../../../configs"):
        """
        Initialize the environment and evaluation environment using the provided configuration.

        :param env_str: The environment string identifier for the Gym environment.
        :param config_path: The path to the configuration files. Default is "../../../configs".
        """
        with initialize(config_path=config_path, version_base="1.2"):
            cfg = compose(config_name="config_" + env_str)

        self.env = make_vec_normalized_env(env_str,
                                           training=True,
                                           norm_obs=cfg.environment.norm_obs,
                                           norm_reward=cfg.environment.norm_reward,
                                           clip_obs=cfg.environment.clip_obs,
                                           clip_reward=cfg.environment.clip_reward,
                                           gamma=cfg.environment.gamma,
                                           epsilon=cfg.environment.epsilon)
        self.eval_env = self.env

    def _config_obj(self, agent_type: str, agent_id: str, env_str: str, config_path: str = "../../../configs"):
        """
        Constructs configuration object for the specified agent type and environment based on the YAML config file.

        :param agent_type: The type of the agent.
        :param agent_id: The identifier of the agent.
        :param env_str: The environment string identifier for the Gym environment.
        :param config_path: The path to the configuration files. Default is "../../../configs".
        :return: The configuration object.
        """
        with initialize(config_path=config_path + "/" + agent_type, version_base="1.2"):
            cfg = compose(config_name="config_" + agent_id + "_" + env_str)

        self.agents_configs[agent_id] = cfg
        return cfg

    def _add_agent_to_df(self, agent_id: str, agent_type: str, hyperparams: dict = None) -> None:
        """
        Add an agent to the dataframe.

        :param agent_id: The identifier of the agent.
        :param agent_type: The type of the agent.
        :param hyperparams: The hyperparameters of the agent. Default is None.
        """
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
        """
        Perform a Wilcoxon rank-sum test comparing the evaluation performance of the agent with a random policy.
        Compares means of undicounted returns.

        :param agent_id: The identifier of the agent.
        """
        tracker = self.metrics_tracker_registry.get_tracker("eval")
        return_history_random = tracker.value_history("return").get("random")
        return_history = tracker.value_history("return").get(agent_id)
        statistic, p_value = ranksums(return_history, return_history_random, alternative='greater')

        if self.verbose > 0:
            logger.info(f"{agent_id} p-value (returns eval): {p_value}")

        self.df.loc[self.df['agent_id'] == agent_id, "p_val_return_eval_>"] = round(p_value, 3)
        self.df.loc[self.df['agent_id'] == agent_id, "W-statistic_return_eval"] = round(statistic, 3)

    def load_data_from_csv(self, data_name: str = "simulation", data_path: str = "../data/experiments") \
            -> 'SimulatorRL':
        self.df = pd.read_csv(data_path + data_name + ".csv")
        return self

    def data_to_csv(self, data_path: str = "../data/experiments") -> 'SimulatorRL':
        self.df.to_csv(data_path + "/" + self.experiment_id + ".csv", index=False)
        return self

    def register_env_str(self, env_str) -> 'SimulatorRL':
        self.env_str = env_str
        return self

    def agent_id_list(self) -> list[str]:
        return list(self.agents_info.keys())

