from abc import ABC, abstractmethod

from torchrl.data import ReplayBuffer

from agent.abstractagent import AbstractAgent
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
        self.models = None
        self.replay_buffer = None

    def set_models(self, models: dict):
        self.models = models
        return self

    def set_replay_buffer(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
        return self

    def valid(self) -> bool:
        return self.models is not None and self.replay_buffer is not None

    @abstractmethod
    def build(self) -> AbstractAgent:
        pass
