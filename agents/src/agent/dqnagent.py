import copy

from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.agent import Agent
from exploration.egreedy import EpsilonGreedy
from util.fetchdevice import fetch_device


class DQNAgent(Agent):
    def __init__(self, model, exploration_policy: EpsilonGreedy):
        super().__init__({"value_network":model, "target_network": copy.deepcopy(model)},
                         memory=ReplayBuffer(storage=LazyTensorStorage(10000, device=fetch_device())))
        self._exploration_policy = exploration_policy
        self._trainer = 