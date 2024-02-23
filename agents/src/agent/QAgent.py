import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.agent import Agent


class QAgent(Agent):
    def __init__(self, model, obs_space, action_space, device):
        """
        Simple implementation of a Q-learning agent with function approximation
        for Q function.
        :param models:
        :param obs_space:
        :param action_space:
        :param device:
        """
        super().__init__({"value_network": model}, obs_space, action_space, device,
                         memory=ReplayBuffer(storage=LazyTensorStorage(1)))

    def add_trajectory(self, trajectory):
        state, action, reward, next_state = trajectory

    def sample_trajectory(self, batch_size=1):
        pass

    def update(self):
        pass

    def policy(self, state):
        pass

    def load_parameters(self):
        pass

    def save_parameters(self):
        pass
