from threading import current_thread

import gymnasium as gym
import torch
import torchrl
from torchrl.modules import EGreedyModule

from agent.agent import Agent
from agent.qagent import QAgent
from exploration.egreedy import EpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from models.linear import LinearModel
from torchrl.data import TensorSpec


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
    def create_agent(agent_type: str, env: gym.Env) -> Agent:
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
            model = ModelFactory.create_model(model_type="linear",
                                              input_size=obs_space.shape[0],
                                              output_size=action_space.n)
            return QAgent(model=model, exploration_policy=EpsilonGreedy(model, action_space))
        elif agent_type == "mlp_q_agent":
            model: torch.nn.Module = ModelFactory.create_model(model_type="mlp",
                                                               input_size=obs_space.shape[0],
                                                               output_size=action_space.n)
            return QAgent(model=model, exploration_policy=EpsilonGreedy(model, action_space))

        raise ValueError("Invalid agent type")
