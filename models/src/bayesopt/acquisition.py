from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior

import gymnasium as gym
from util.fetchdevice import fetch_device


def append_actions(state: torch.Tensor, action_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Append actions to a state tensor.

    :param state: State tensor.
    :param action_size: Number of possible actions.
    :param device: Device to perform operations on. If None, fetch the available device.
    :return: State-action pairs tensor.
    """
    if device is None:
        device = fetch_device()

    repeated_state = torch.stack([state] * action_size)
    actions = torch.arange(action_size).to(device)
    state_action_pairs = torch.cat([repeated_state, actions.unsqueeze(1)], dim=-1)

    return state_action_pairs


class GPActionSelector(ABC):
    """
    Abstract base class for Gaussian Process action selectors.
    """

    @abstractmethod
    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Select an action based on the GPQ model and state.

        :param gpq_model: Gaussian Process model.
        :param state_tensor: State tensor.
        :return: Selected action tensor.
        """
        pass

    def update(self) -> None:
        """
        Update the action selector.
        """
        pass


class ThompsonSampling(GPActionSelector):
    """
    Thompson Sampling action selector.
    """

    def __init__(self, action_size: int, observation_noise: bool = False):
        """
        Constructor for ThompsonSampling.

        :param action_size: Number of possible actions.
        :param observation_noise: Whether to include observation noise in the sampling.
        """
        self.action_size = action_size
        self.observation_noise = observation_noise

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform Thompson Sampling to select an action.
        Select the action with the highest sample Q-value.

        :param gpq_model: Gaussian Process model.
        :param state_tensor: State tensor.
        :return: Selected action tensor.
        """
        state_action_pairs = append_actions(state_tensor, self.action_size)
        posterior_distribution: GPyTorchPosterior = gpq_model.posterior(
            state_action_pairs, observation_noise=self.observation_noise)
        sampled_q_values = posterior_distribution.rsample()
        best_action = torch.argmax(sampled_q_values, dim=1)
        return best_action


class UpperConfidenceBound(GPActionSelector):
    """
    Upper Confidence Bound action selector.
    """

    def __init__(self, action_size: int, beta: float = 1.5, observation_noise: bool = False):
        """
        Constructor for UpperConfidenceBound.

        :param action_size: Number of possible actions.
        :param beta: Confidence level parameter.
        :param observation_noise: Whether to include observation noise in the sampling.
        """
        self.action_size = action_size
        self.beta = beta
        self.observation_noise = observation_noise

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform Upper Confidence Bound action selection.

        :param gpq_model: Gaussian Process model.
        :param state_tensor: State tensor.
        :return: Selected action tensor.
        """
        state_action_pairs = append_actions(state_tensor, self.action_size)
        posterior_distribution: GPyTorchPosterior = gpq_model.posterior(
            state_action_pairs, observation_noise=self.observation_noise)
        confident_q_values = posterior_distribution.mean + self.beta * torch.sqrt(posterior_distribution.variance)
        confident_q_values = confident_q_values.unsqueeze(0)
        best_action = torch.argmax(confident_q_values, dim=1)
        return best_action


class GPEpsilonGreedy(GPActionSelector):
    """
    Epsilon-Greedy action selector for Gaussian Process models.
    """

    def __init__(self, action_space: gym.Space, eps_init: float = 1.0, eps_end: float = 0.1,
                 annealing_num_steps: int = 3500, observation_noise: bool = False):
        """
        Constructor for GPEpsilonGreedy.

        :param action_space: The action space of the environment.
        :param eps_init: Initial epsilon value for epsilon-greedy policy.
        :param eps_end: Final epsilon value for epsilon-greedy policy.
        :param annealing_num_steps: Number of steps over which epsilon is annealed.
        :param observation_noise: Whether to include observation noise in the sampling.
        """
        self._action_space = action_space
        self._epsilon = eps_init
        self._epsilon_target = eps_end
        self._epsilon_delta = (eps_init - eps_end) / annealing_num_steps
        self._observation_noise = observation_noise

    def update(self) -> None:
        """
        Update the epsilon value.
        """
        if self._epsilon > self._epsilon_target:
            self._epsilon = max(self._epsilon_target, self._epsilon - self._epsilon_delta)

    def action(self, gpq_model: GPyTorchModel, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform epsilon-greedy action selection.

        :param gpq_model: Gaussian Process model.
        :param state_tensor: State tensor.
        :return: Selected action tensor.
        """
        if np.random.uniform(0, 100) >= self._epsilon * 100:
            state_action_pairs = append_actions(state_tensor, self._action_space.n)
            posterior_distribution: GPyTorchPosterior = gpq_model.posterior(
                state_action_pairs, observation_noise=self._observation_noise)
            q_values = posterior_distribution.mean.unsqueeze(0)
            best_action = torch.argmax(q_values, dim=1)
            return best_action

        return torch.tensor([[self._action_space.sample()]], device=fetch_device())
