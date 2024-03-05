from overrides import overrides
from torchrl.data import LazyTensorStorage, ReplayBuffer

from agent.abstractagent import AbstractAgent
from agent.qagent import QAgent
from builders.valueagentbuilder import ValueAgentBuilder
from trainers.rltrainer import RLTrainer
from util.fetchdevice import fetch_device


class QAgentBuilder(ValueAgentBuilder):
    def __init__(self):
        super().__init__()
        self.set_replay_buffer()

    def init_trainer(self) -> RLTrainer:
        pass

    def init_replay_buffer(self) -> None:
        # Note: Classical Q-learning does not use a ReplayBuffer, so we set
        # the ReplayBuffer size to 1.
        return ReplayBuffer(storage=LazyTensorStorage(1, device=fetch_device()))

    def build(self) -> AbstractAgent:
        if self.valid():
            return QAgent(self)
        else:
            raise ValueError("Not all required attributes are set.")
