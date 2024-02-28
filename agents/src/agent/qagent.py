import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.agent import Agent
from agent.valueagent import ValueAgent
from util.fetchdevice import fetch_device
from trainers.qtrainer import QTrainer


class QAgent(ValueAgent):
    def __init__(self, model, exploration_policy,
                 replay_buffer=ReplayBuffer(storage=LazyTensorStorage(1, device=fetch_device()))):
        """
        Simple implementation of a Q-learning agent with function approximation
        for Q function.
        :param model: for function approximation.
        :param exploration_policy: i.e., epsilon greedy.
        """
        # Note: Classical Q-learning does not use a ReplayBuffer, so we set
        # the ReplayBuffer size to 1.
        super().__init__({"value_network": model},
                         memory=replay_buffer,
                         exploration_policy=exploration_policy,
                         trainer=QTrainer(model=model, batch_size=1, buf=replay_buffer))
