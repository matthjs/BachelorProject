from threading import current_thread

import gymnasium as gym
from stable_baselines3 import DQN, PPO

from agent.abstractagent import AbstractAgent
from agent.gaussianprocessdpagent import GaussianProcessDPAgent
from agent.gpqagent import GPQAgent
from agent.gpsarsaagent import GPSarsaAgent
from agent.randomagent import RandomAgent
from agent.sbadapter import StableBaselinesAdapter
from bachelorproject.configobject import Config
from builders.dqnagentbuilder import DQNAgentBuilder
from builders.qagentbuilder import QAgentBuilder
from torchrl.data import LazyTensorStorage


class AgentFactory:
    """
    Naive factory method implementation for RL agent creation.
    """

    @staticmethod
    def set_thread_id(agent_type: str) -> None:
        """
        Set the thread ID for the current thread.
        :param agent_type: The type of agent.
        """
        thr = current_thread()
        thr.name = agent_type + "_thread_" + str(thr.ident)

    @staticmethod
    def create_agent_configured(agent_type: str, env_str: str, cfg) -> tuple[AbstractAgent, dict]:
        """
        Create an RL agent based on the provided configuration.
        WARNING: THIS FUNCTION IS A BIT OF A MESS.
        :param agent_type: The type of agent to create.
        :param env_str: The name of the Gym environment.
        :param cfg: The configuration for the agent.
        :return: A tuple containing the created agent and its configuration.
        """
        env = gym.make(env_str)

        if agent_type in ["gpq_agent", "gpsarsa_agent"]:
            Config.initialize_config_object(cfg)

        if agent_type == "gpq_agent":
            result = {
                'gp_model_str': cfg.model.gp_model_str,
                'discount_factor': cfg.model.discount_factor,
                'batch_size': cfg.model.batch_size,
                'replay_buffer_size': cfg.model.replay_buffer_size,
                'exploring_starts': cfg.model.exploring_starts,
                'max_dataset_size': cfg.model.max_dataset_size,
                'kernel_type': cfg.model.kernel_type,
                'sparsification_threshold': eval(cfg.model.sparsification_threshold),
                'strategy': cfg.model.strategy,
                'posterior_observation_noise': cfg.model.posterior_observation_noise,
                'num_inducing_points': cfg.model.num_inducing_points
            }

            if cfg.model.gp_model_str == "deep_gp":
                result.update(cfg.fitting)
                result.update({"dgp_hidden_layers_config": cfg.dgp_hidden_layers_config})

            if cfg.model.strategy == "upper_confidence_bound":
                result.update({'ucb_beta': cfg.exploration.ucb_beta})

            if cfg.model.strategy == "epsilon_greedy":
                result.update({'gp_e_greedy_steps': cfg.exploration.gp_e_greedy_steps})
            return GPQAgent(
                gp_model_str=cfg.model.gp_model_str,
                env=env,
                discount_factor=cfg.model.discount_factor,
                batch_size=cfg.model.batch_size,
                replay_buffer_size=cfg.model.replay_buffer_size,
                exploring_starts=cfg.model.exploring_starts,
                max_dataset_size=cfg.model.max_dataset_size,
                kernel_type=cfg.model.kernel_type,
                kernel_args=cfg.model.kernel,
                sparsification_threshold=eval(cfg.model.sparsification_threshold),
                strategy=cfg.model.strategy,
                posterior_observation_noise=cfg.model.posterior_observation_noise,
                num_inducing_points=cfg.model.num_inducing_points
            ), result
        elif agent_type == "gpsarsa_agent":
            result = {
                'gp_model_str': cfg.model.gp_model_str,
                'discount_factor': cfg.model.discount_factor,
                'batch_size': cfg.model.batch_size,
                'replay_buffer_size': cfg.model.replay_buffer_size,
                'exploring_starts': cfg.model.exploring_starts,
                'max_dataset_size': cfg.model.max_dataset_size,
                'kernel_type': cfg.model.kernel_type,
                'sparsification_threshold': eval(cfg.model.sparsification_threshold),
                'strategy': cfg.model.strategy,
                'posterior_observation_noise': cfg.model.posterior_observation_noise,
                'num_inducing_points': cfg.model.num_inducing_points
            }

            if cfg.model.gp_model_str == "deep_gp":
                result.update(cfg.fitting)
                result.update({"dgp_hidden_layers_config": cfg.dgp_hidden_layers_config})

            if cfg.model.strategy == "upper_confidence_bound":
                result.update({'ucb_beta': cfg.exploration.ucb_beta})

            if cfg.model.strategy == "epsilon_greedy":
                result.update({'gp_e_greedy_steps': cfg.exploration.gp_e_greedy_steps})
            return GPSarsaAgent(
                gp_model_str=cfg.model.gp_model_str,
                env=env,
                discount_factor=cfg.model.discount_factor,
                batch_size=cfg.model.batch_size,
                replay_buffer_size=cfg.model.replay_buffer_size,
                exploring_starts=cfg.model.exploring_starts,
                max_dataset_size=cfg.model.max_dataset_size,
                kernel_type=cfg.model.kernel_type,
                kernel_args=cfg.model.kernel,
                sparsification_threshold=eval(cfg.model.sparsification_threshold),
                strategy=cfg.model.strategy,
                posterior_observation_noise=cfg.model.posterior_observation_noise,
                num_inducing_points=cfg.model.num_inducing_points
            ), result
        elif agent_type == "sb_dqn":
            return StableBaselinesAdapter(
                DQN(
                    policy=cfg.model.policy,
                    env=env,
                    learning_rate=cfg.model.learning_rate,
                    batch_size=cfg.model.batch_size,
                    buffer_size=cfg.model.buffer_size,
                    learning_starts=cfg.model.learning_starts,
                    gamma=cfg.model.gamma,
                    target_update_interval=cfg.model.target_update_interval,
                    train_freq=cfg.model.train_freq,
                    gradient_steps=cfg.model.gradient_steps,
                    exploration_fraction=cfg.model.exploration_fraction,
                    exploration_final_eps=cfg.model.exploration_final_eps,
                    policy_kwargs=eval(cfg.model.policy_kwargs)
                )
            ), {
                'policy': cfg.model.policy,
                'learning_rate': cfg.model.learning_rate,
                'batch_size': cfg.model.batch_size,
                'buffer_size': cfg.model.buffer_size,
                'learning_starts': cfg.model.learning_starts,
                'gamma': cfg.model.gamma,
                'target_update_interval': cfg.model.target_update_interval,
                'train_freq': cfg.model.train_freq,
                'gradient_steps': cfg.model.gradient_steps,
                'exploration_fraction': cfg.model.exploration_fraction,
                'exploration_final_eps': cfg.model.exploration_final_eps,
                'policy_kwargs': eval(cfg.model.policy_kwargs)
            }
        elif agent_type == "sb_ppo":
            return StableBaselinesAdapter(
                PPO(
                    policy=cfg.model.policy,
                    env=env,
                    n_steps=cfg.model.n_steps,
                    batch_size=cfg.model.batch_size,
                    gae_lambda=cfg.model.gae_lambda,
                    gamma=cfg.model.gamma,
                    n_epochs=cfg.model.n_epochs,
                    ent_coef=cfg.model.ent_coef,
                    learning_rate=cfg.model.learning_rate,
                    clip_range=cfg.model.clip_range
                )
            ), {
                'policy': cfg.model.policy,
                'n_steps': cfg.model.n_steps,
                'batch_size': cfg.model.batch_size,
                'gae_lambda': cfg.model.gae_lambda,
                'gamma': cfg.model.gamma,
                'n_epochs': cfg.model.n_epochs,
                'ent_coef': cfg.model.ent_coef,
                'learning_rate': cfg.model.learning_rate,
                'clip_range': cfg.model.clip_range
            }
        elif agent_type == "linear_q_agent":
            return (QAgentBuilder()
                    .set_env(env)
                    .set_replay_buffer_size(cfg.model.replay_buffer_size)
                    .set_batch_size(cfg.model.batch_size)
                    .set_value_model_type_str(cfg.model.model_type)
                    .set_buffer_storage_type(LazyTensorStorage)
                    .set_annealing_num_steps(cfg.model.annealing_num_steps)
                    .set_learning_rate(cfg.model.learning_rate)
                    .set_discount_factor(cfg.model.discount_factor)
                    .build()), {
                'value_model': cfg.model.model_type,
                'batch_size': cfg.model.batch_size,
                'replay_buffer_size': cfg.model.replay_buffer_size,
                'annealing_num_steps': cfg.model.annealing_num_steps,
                'learning_rate': cfg.model.learning_rate,
                'discount_factor': cfg.model.learning_rate
            }
        elif agent_type == "random":
            return RandomAgent(env), {}

        raise ValueError("Unsupported agent type")

    @staticmethod
    def create_agent(agent_type: str, env_str: str) -> AbstractAgent:
        """
        Factory method for Agent creation.
        NOTE: This factory function assumes continuous state spaces and
        discrete action spaces.
        :param env:
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        env = gym.make(env_str)
        obs_space = env.observation_space
        action_space = env.action_space

        AgentFactory.set_thread_id(agent_type)

        if agent_type == "linear_q_agent":
            return (QAgentBuilder()
                    .set_env(env)
                    .set_replay_buffer_size(1)
                    .set_batch_size(1)
                    .set_value_model_type_str("linear")
                    .set_buffer_storage_type(LazyTensorStorage)
                    .set_annealing_num_steps(2000)
                    .set_learning_rate(0.001)
                    .set_discount_factor(0.9)
                    .build())
        elif agent_type == "mlp_q_agent":
            return (QAgentBuilder()
                    .set_env(env)
                    .set_replay_buffer_size(1)
                    .set_batch_size(1)
                    .set_value_model_type_str("mlp")
                    .set_annealing_num_steps(2000)
                    .set_learning_rate(0.001)
                    .set_discount_factor(0.9)
                    .build())
        elif agent_type == "mlp_dqn_agent":
            return (DQNAgentBuilder()
                    .set_env(env)
                    .set_replay_buffer_size(5000)
                    .set_batch_size(32)
                    .set_value_model_type_str("mlp")
                    .set_annealing_num_steps(2000)
                    .set_learning_rate(0.001)
                    .set_discount_factor(0.9)
                    .set_param_copying(20)
                    .build())
        elif agent_type == "gaussian_dp_agent":
            return GaussianProcessDPAgent(env)
        elif agent_type == "gpq_agent":
            return GPQAgent(gp_model_str="exact_gp",
                            env=env,
                            discount_factor=0.99,
                            batch_size=64,
                            replay_buffer_size=64,
                            exploring_starts=1000,
                            max_dataset_size=10000,
                            sparsification_treshold=0.1)
        elif agent_type == "gpsarsa_agent":
            return GPSarsaAgent(gp_model_str="exact_gp",
                                env=env,
                                discount_factor=0.99,
                                batch_size=64,
                                replay_buffer_size=64,
                                exploring_starts=1000,
                                max_dataset_size=10000)
        elif agent_type == "random":
            return RandomAgent(env)

        raise ValueError("Invalid agent type")
