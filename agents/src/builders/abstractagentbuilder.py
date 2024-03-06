import gymnasium as gym
from abc import ABC, abstractmethod


class AbstractAgentBuilder(ABC):
    """
    Builder pattern using inner class. This is necessary as agents need to have
    various components instantiated in a specific order.
    """

    def __init__(self):
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
    def build(self):
        pass
