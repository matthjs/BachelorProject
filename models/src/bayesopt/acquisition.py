from abc import ABC, abstractmethod

import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior

import gymnasium as gym
from util.fetchdevice import fetch_device


def append_actions(state: torch.Tensor, action_size: int, device=None) -> torch.Tensor:
    """
    Append actions to a state tensor.
    """
    if device is None:
        device = fetch_device()

    repeated_state = torch.stack([state] * action_size)
    actions = torch.arange(action_size).to(device)
    state_action_pairs = torch.cat([repeated_state, actions.unsqueeze(1)], dim=-1)

    return state_action_pairs


def simple_thompson_action_sampler(gpq_model: GPyTorchModel, state_tensor: torch.Tensor, action_size: int) -> torch.Tensor:
    """
    Thompson sampling for discrete action spaces.
    """
    state_action_pairs = append_actions(state_tensor, action_size)
    posterior_distribution: GPyTorchPosterior = gpq_model.posterior(state_action_pairs)
    sampled_q_values = posterior_distribution.rsample()
    best_action = torch.argmax(sampled_q_values, dim=1)
    return best_action


def upper_confidence_bound_selector(gpq_model: GPyTorchModel, state_tensor: torch.Tensor, action_size: int, beta=1) -> torch.Tensor:
    """
    Upper confidence bound action selection.
    """
    state_action_pairs = append_actions(state_tensor, action_size)
    posterior_distribution = gpq_model.posterior(state_action_pairs)
    confident_q_values = posterior_distribution.mean + beta * torch.sqrt(posterior_distribution.variance)
    confident_q_values = confident_q_values.unsqueeze(0)
    best_action = torch.argmax(confident_q_values, dim=1)
    return best_action


class GPActionSelector(ABC):
    """
    Abstract base class for Gaussian Process action selectors.
    """

    @abstractmethod
    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Select an action based on the GPQ model and state.
        """
        pass

    def update(self):
        """
        Update the action selector.
        """
        pass


class ThompsonSampling(GPActionSelector):
    """
    Thompson Sampling action selector.
    """

    def __init__(self, action_size: int):
        self.action_size = action_size

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform Thompson Sampling to select an action.
        """
        state_action_pairs = append_actions(state_tensor, self.action_size)
        posterior_distribution: GPyTorchPosterior = gpq_model.posterior(state_action_pairs, observation_noise=True)
        sampled_q_values = posterior_distribution.rsample()
        best_action = torch.argmax(sampled_q_values, dim=1)
        return best_action


class UpperConfidenceBound(GPActionSelector):
    """
    Upper Confidence Bound action selector.
    """

    def __init__(self, action_size, beta=1):
        self.action_size = action_size
        self.beta = beta

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform Upper Confidence Bound action selection.
        """
        state_action_pairs = append_actions(state_tensor, self.action_size)
        posterior_distribution = gpq_model.posterior(state_action_pairs, observation_noise=True)
        confident_q_values = posterior_distribution.mean + self.beta * torch.sqrt(posterior_distribution.variance)
        confident_q_values = confident_q_values.unsqueeze(0)
        best_action = torch.argmax(confident_q_values, dim=1)
        return best_action


class GPEpsilonGreedy(GPActionSelector):
    """
    Epsilon-Greedy action selector for Gaussian Process models.
    """

    def __init__(self, action_space: gym.Space, eps_init=1.0, eps_end=0.1, annealing_num_steps=5000):
        self._action_space = action_space
        self._epsilon = eps_init
        self._epsilon_target = eps_end
        self._epsilon_delta = (1 - self._epsilon_target) / annealing_num_steps

    def update(self) -> None:
        """
        Update the epsilon value.
        """
        if self._epsilon > self._epsilon_target:
            self._epsilon = max(self._epsilon_target, self._epsilon - self._epsilon_delta)

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform epsilon-greedy action selection.
        """
        if np.random.uniform(0, 100) >= self._epsilon * 100:
            state_action_pairs = append_actions(state_tensor, self._action_space.n)
            posterior_distribution = gpq_model.posterior(state_action_pairs, observation_noise=True)
            best_action, _ = torch.argmax(posterior_distribution.mean, dim=1)
            return best_action

        return torch.tensor([[self._action_space.sample()]], device=fetch_device())
