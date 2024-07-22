import gymnasium as gym

from stable_baselines3 import DQN, PPO
from agent.abstractagent import AbstractAgent
from agent.gpqagent import GPQAgent
from agent.gpsarsaagent import GPSarsaAgent
from agent.randomagent import RandomAgent
from agent.sbadapter import StableBaselinesAdapter
from bachelorproject.configobject import Config


# noinspection DuplicatedCode
class AgentFactory:
    """
    Naive factory method implementation for RL agent creation.
    """

    @staticmethod
    def create_agent_configured(agent_type: str, env, cfg) -> tuple[AbstractAgent, dict]:
        """
        Create an RL agent based on the provided configuration.
        WARNING: THIS FUNCTION IS A MESS AND DOES NOT REFLECT MY OWN PERSONAL STANDARD ON
        SOFTWARE QUALITY.
        :param agent_type: The type of agent to create.
        :param env: A (wrapped) Gymnasium environment.
        :param cfg: The configuration for the agent.
        :return: A tuple containing the created agent and its configuration.
        """

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
        elif agent_type == "random":
            return RandomAgent(env), {}

        raise ValueError("Unsupported agent type")

    @staticmethod
    def create_random_agent(env_str: str) -> AbstractAgent:
        """
        Create random agent.
        :param env_str:
        :return: an object of type Agent.
        """
        env = gym.make(env_str)
        return RandomAgent(env)
