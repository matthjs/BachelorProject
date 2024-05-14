import pandas as pd
import gymnasium as gym
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from agentfactory.agentfactory import AgentFactory
from callbacks.abstractcallback import AbstractCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.sbcallbackadapter import StableBaselinesCallbackAdapter
from metricstracker.metricstrackerregistry import MetricsTrackerRegistry
from hydra import compose, initialize
import cloudpickle


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
        self.agents_configs["random"] = None

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

    def plot_any_plottable_data(self, plot_dir: str = "../plots/") -> 'SimulatorRL':
        tracker = self.metrics_tracker_registry.get_tracker("train")
        tracker.plot_metric(metric_name="return",
                            plot_path=plot_dir + self.env_str,
                            title=self.env_str + "_" + list(self.agents.keys()).__str__())
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

    def save(self, agent_id_list: list[str] = None) -> 'SimulatorRL':
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        # I am being really lazy here. Try and see if cloudpickle suffices.
        # Even though for StableBaselines agents there is the .save() .load methods.
        print(self.agents_configs)

        return self



    def load(self, agent_id_list: list[str] = None) -> 'SimulatorRL':


        return self

    def train_agents(self,
                     num_episodes: int,
                     agent_id_list: list[str] = None,
                     callbacks: list[AbstractCallback] = None,
                     concurrent=False,
                     logging=True) \
            -> 'SimulatorRL':
        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if callbacks is None:
            callbacks = [RewardCallback()]

        if not concurrent:
            for agent_id in agent_id_list:
                if agent_id not in self.agents:
                    raise ValueError("agent_id not in agents dictionary")

                if self.agents[agent_id].is_stable_baselines_wrapper():
                    self._stable_baselines_train(agent_id, num_episodes, callbacks, logging)
                else:
                    self._agent_env_interaction_gym("train", agent_id, num_episodes, callbacks, logging)
        else:
            raise NotImplementedError("OOPS I did not implement this yet my bad.")

        return self

    def evaluate_agents(self,
                        eval_env_str: str,
                        num_episodes: int,
                        agent_id_list: list[str] = None,
                        callbacks: list[AbstractCallback] = None,
                        concurrent=False,
                        logging=True) -> 'SimulatorRL':
        if logging:
            print("Evaluating...")

        if callbacks is None:
            callbacks = [RewardCallback()]

        # Number of episodes should probably be in config but ah well.
        if agent_id_list is None:
            agent_id_list = list(self.agents.keys())  # do all agents if agent_id_list not specified.

        if not concurrent:
            for agent_id in agent_id_list:
                if agent_id not in self.agents:
                    raise ValueError("agent_id not in agents dictionary")

                self._agent_env_interaction_gym("eval", agent_id, num_episodes, callbacks, logging)

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

    def _agent_env_interaction_gym(self,
                                   mode: str,
                                   agent_id: str,
                                   num_episodes: int,
                                   callbacks: list[AbstractCallback],
                                   logging) -> None:
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
                mode,
                agent,
                agent_id,
                self.agents_configs[agent_id],
                self.df,
                self.metrics_tracker_registry,
                logging
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
                                callbacks: list[AbstractCallback],
                                logging=True) -> None:
        agent = self.agents[agent_id]
        if not agent.is_stable_baselines_wrapper():
            raise AttributeError("This function should only be run on wrapped StableBaselines agents")

        sb_callbacks = [StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=0)]

        for callback in callbacks:
            callback.init_callback(
                "train",
                agent,
                agent_id,
                self.agents_configs[agent_id],
                self.df,
                self.metrics_tracker_registry,
                logging
            )
            sb_callbacks.append(StableBaselinesCallbackAdapter(callback))

        model = agent.stable_baselines_unwrapped()

        model.learn(total_timesteps=int(5e4), callback=sb_callbacks)
