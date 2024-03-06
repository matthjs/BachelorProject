from threading import current_thread

import gymnasium as gym

from agent.abstractagent import AbstractAgent
from builders.dqnagentbuilder import DQNAgentBuilder
from builders.qagentbuilder import QAgentBuilder
from torchrl.data import LazyTensorStorage


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """
    @staticmethod
    def set_thread_id(agent_type: str) -> None:
        thr = current_thread()
        thr.name = agent_type + "_thread_" + str(thr.ident)

    @staticmethod
    def create_agent(agent_type: str, env: gym.Env) -> AbstractAgent:
        """
        Factory method for Agent creation.
        NOTE: This factory function assumes continuous state spaces and
        discrete action spaces.
        :param env:
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
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
                    .set_replay_buffer_size(32)
                    .set_batch_size(5000)
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

        raise ValueError("Invalid agent type")