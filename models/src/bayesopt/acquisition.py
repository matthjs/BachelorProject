from abc import ABC, abstractmethod

import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior

import gymnasium as gym
from util.fetchdevice import fetch_device

"""
Note to self: botorch have prebuild acquisition functions, but I am not sure have applicable
they are in a RL setting.
"""


def append_actions(state: torch.tensor, action_size: int, device=None) -> torch.tensor:
    """
    Append to a state vector (s1, s2, ..., s_n) actions such that we have
    a batch of tensors of the form: ((s1, s2, ..., s_n, a_1),
                                     (s1, s2, ..., s_n, a_2),
                                     ...
                                     (s1, s2, ..., s_n, a_m)) where  m = num of actions.
    :param state: state tensor.
    :param action_size: number of actions.
    :param device: GPU or CPU.
    :return: batched state action tensor as described above.
    """
    if device is None:
        device = fetch_device()

    # Repeat the state vector for each action
    repeated_state = torch.stack([state] * action_size)

    # Create a tensor for actions ranging from 0 to action_size - 1
    actions = torch.arange(action_size).to(device)

    # Concatenate the repeated state vectors with the actions
    state_action_pairs = torch.cat([repeated_state, actions.unsqueeze(1)], dim=-1)

    return state_action_pairs


def simple_thompson_action_sampler(gpq_model: GPyTorchModel,
                                   state_tensor: torch.tensor,
                                   action_size: int) -> torch.tensor:
    """
    Thompson sampling for discrete action spaces.
    Assumes last dimension is action dimension.
    :param gpq_model: GP regression model for Q : S x A -> R
    :param state_tensor: a state variable.
    :param action_size: action encoding {0, 1, ..., n - 1}
    :return: best action according to sample q-function
    """
    state_action_pairs = append_actions(state_tensor, action_size)

    posterior_distribution: GPyTorchPosterior = gpq_model.posterior(state_action_pairs)
    sampled_q_values = posterior_distribution.rsample()

    best_action = torch.argmax(sampled_q_values, dim=1)

    return best_action


def upper_confidence_bound_selector(gpq_model: GPyTorchModel,
                                    state_tensor: torch.tensor,
                                    action_size: int,
                                    beta=1) -> torch.tensor:
    state_action_pairs = append_actions(state_tensor, action_size)
    posterior_distribution = gpq_model.posterior(state_action_pairs)
    confident_q_values = posterior_distribution.mean + beta * torch.sqrt(posterior_distribution.variance)
    confident_q_values = confident_q_values.unsqueeze(0)

    best_action = torch.argmax(confident_q_values, dim=1)

    return best_action


class GPActionSelector(ABC):
    @abstractmethod
    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.tensor) -> torch.tensor:
        pass

    def update(self):
        pass


class ThompsonSampling(GPActionSelector):
    def __init__(self, action_size: int):
        self.action_size = action_size

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.tensor) -> torch.tensor:
        """
        Thompson sampling for discrete action spaces.
        Assumes last dimension is action dimension.
        :param gpq_model: GP regression model for Q : S x A -> R
        :param state_tensor: a state variable.
        :param action_size: action encoding {0, 1, ..., n - 1}
        :return: best action according to sample q-function
        """
        state_action_pairs = append_actions(state_tensor, self.action_size)

        posterior_distribution: GPyTorchPosterior = gpq_model.posterior(state_action_pairs)
        sampled_q_values = posterior_distribution.rsample()

        # shape [1, 2, 1] 1 sample of (2, 1)

        best_action = torch.argmax(sampled_q_values, dim=1)

        return best_action


class UpperConfidenceBound(GPActionSelector):
    def __init__(self, action_size, beta=1):
        self.action_size = action_size
        self.beta = beta

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.tensor) -> torch.tensor:
        state_action_pairs = append_actions(state_tensor, self.action_size)
        posterior_distribution = gpq_model.posterior(state_action_pairs)

        confident_q_values = posterior_distribution.mean + self.beta * torch.sqrt(posterior_distribution.variance)
        confident_q_values = confident_q_values.unsqueeze(0)

        best_action = torch.argmax(confident_q_values, dim=1)

        return best_action


class GPEpsilonGreedy(GPActionSelector):
    """
    Class supporting epsilon greedy action selection with decay.
    """

    def __init__(self,
                 action_space: gym.Space,
                 eps_init=1.0,
                 eps_end=0.1,
                 annealing_num_steps=5000):
        """
        :param model: action-value function estimate.
        :param action_space:
        :param eps_init: initial epsilon value. default: 1.0
        :param eps_end: final epsilon value. default: 0.1
        :param annealing_num_steps: number of steps it will take for epsilon to reach the eps_end value.
        Defaults to 1000.
        """
        self._action_space = action_space
        self._epsilon = eps_init
        self._epsilon_target = eps_end
        self._epsilon_delta = (1 - self._epsilon_target) / annealing_num_steps

    def update(self) -> None:
        """
        Updates epsilon schedule [linear]
        """
        if self._epsilon > self._epsilon_target:
            self._epsilon = max(self._epsilon_target, self._epsilon - self._epsilon_delta)

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.tensor) -> torch.tensor:
        if np.random.uniform(0, 100) >= self._epsilon * 100:
            state_action_pairs = append_actions(state_tensor, self._action_space.n)
            posterior_distribution = gpq_model.posterior(state_action_pairs)
            best_action = torch.argmax(posterior_distribution.mean, dim=1)
            return best_action

        return torch.tensor([[self._action_space.sample()]], device=fetch_device())
