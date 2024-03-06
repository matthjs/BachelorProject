import gymnasium as gym
from abc import ABC, abstractmethod

from torchrl.data import ReplayBuffer

from agent.abstractagent import AbstractAgent
from trainers.rltrainer import RLTrainer
from util.fetchdevice import fetch_device


class EnvInfo:
    """
    Inner class for recording game information.
    """

    def __init__(self):
        self.done = False
        self.info = None

    def set_done(self, done):
        self.done = done

    def set_info(self, info):
        self.info = info


class BaseAgentBuilder(ABC):
    """
    Builder pattern using inner class. This is necessary as agents need to have
    various components instantiated in a specific order.
    """

    def __init__(self):
        self.device = fetch_device()
        self.env_info = EnvInfo()
        self.models = {}
        self.state_space = None
        self.action_space = None
        self.env = None

    def set_state_space(self, state_space):
        self.state_space = state_space
        return self

    def set_action_space(self, action_space):
        self.action_space = action_space
        return self

    def set_env(self, env: gym.Env):
        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space
        return self

    def valid(self) -> bool:
        return self.action_space is not None and self.state_space is not None

    @abstractmethod
    def build(self) -> AbstractAgent:
        pass
