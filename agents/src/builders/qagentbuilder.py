from overrides import overrides
from torchrl.data import LazyTensorStorage, ReplayBuffer

from agent.abstractagent import AbstractAgent
from agent.qagent import QAgent
from builders.valueagentbuilder import ValueAgentBuilder
from exploration.egreedy import EpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from trainers.qtrainer import QTrainer
from trainers.rltrainer import RLTrainer
from util.fetchdevice import fetch_device


class QAgentBuilder(ValueAgentBuilder):
    def __init__(self):
        super().__init__()

    def build(self) -> AbstractAgent:
        if self.valid():
            return QAgent(self)
        else:
            raise ValueError("Not all required attributes are set.")
