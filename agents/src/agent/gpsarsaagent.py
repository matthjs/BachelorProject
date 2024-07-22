import gymnasium as gym
import numpy as np
import torch

from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from typing import Any
from agent.abstractagent import AbstractAgent
from bayesopt.bayesianoptimizer_rl import BayesianOptimizerRL
from trainers.gpsarsatrainer import GPSarsaTrainer
from util.fetchdevice import fetch_device
from util.processstate import process_state


class GPSarsaAgent(AbstractAgent):
    """
    One advantage of the SARSA agent is that it is actually compatible with continuous action spaces
    provided you do not have greedy action selection.
    """

    def __init__(self,
                 gp_model_str: str,
                 env: gym.Env,
                 discount_factor: float,
                 batch_size: int,
                 replay_buffer_size: int,  # batch_size == replay_buffer_size
                 exploring_starts: int,
                 max_dataset_size: int,
                 kernel_type: str,
                 kernel_args: Any = None,  # Kernel args are not really used.
                 strategy: str = "thompson_sampling",
                 posterior_observation_noise: bool = False,
                 num_inducing_points: int = 128) -> None:
        """
        GPSarsaAgent constructor.

        :param gp_model_str: The type of GP model to use.
        :param env: The environment.
        :param discount_factor: The discount factor.
        :param batch_size: The batch size.
        :param replay_buffer_size: The size of the replay buffer.
        :param exploring_starts: Number of exploring starts.
        :param max_dataset_size: The maximum dataset size.
        :param kernel_type: The type of kernel to be used.
        :param kernel_args: Kernel arguments (not really used).
        :param strategy: Strategy to use for action selection.
        :param posterior_observation_noise: Whether to add observation noise to the posterior.
        :param num_inducing_points: Number of inducing points for the variational strategy.
        """
        super(GPSarsaAgent, self).__init__({}, env.observation_space, env.action_space)
        self._training = True

        self._exploration_policy = BayesianOptimizerRL(
            model_str=gp_model_str,
            max_dataset_size=max_dataset_size,
            exploring_starts=exploring_starts,
            action_space=env.action_space,
            kernel_type=kernel_type,
            kernel_args=kernel_args,
            state_space=env.observation_space,
            strategy=strategy,
            posterior_observation_noise=posterior_observation_noise,
            num_inducing_points=num_inducing_points
        )

        # NOTE: The ReplayBuffer is used as temporary storage for a batch of trajectories. Note for sampling
        # as in DQN.
        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=replay_buffer_size, device=fetch_device()),
            sampler=SamplerWithoutReplacement()
        )

        self._batch_counter = 0
        self._batch_size = batch_size

        self._trainer = GPSarsaTrainer(
            self._exploration_policy,
            batch_size=batch_size,
            buf=self._replay_buffer,
            discount_factor=discount_factor
        )

        self._hyperparameters = {
            'gp_model': gp_model_str,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            'exploring_starts': exploring_starts,
            'max_dataset_size': max_dataset_size,
            'kernel_type': kernel_type,
        }

    def update(self) -> None:
        """
        Update the agent.
        """
        if self._training:
            if self._batch_counter >= self._batch_size:
                self._trainer.train()
                self._batch_counter = 0
            self._batch_counter += 1

    def updatable(self) -> bool:
        """
        Check if the agent is ready for update.

        :return: True if the agent is ready, False otherwise.
        """
        return self._training and self._exploration_policy.exploring_starts() <= 0 and self._batch_counter >= self._batch_size

    # noinspection DuplicatedCode
    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the replay buffer.

        :param trajectory: Tuple containing (state, action, reward, next_state).
        """
        state, action, reward, next_state = trajectory
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.double)
        action_t = torch.as_tensor(action, device=self.device)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.double)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.double)

        # To ensure compatibility with off-policy algorithms, run the policy one more time here.
        next_action = self.policy(next_state)
        next_action_t = torch.as_tensor(next_action, device=self.device)

        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t, next_action_t))

    def policy(self, state: np.ndarray) -> int:
        """
        Get the policy for the given state.

        :param state: The state.
        :return: The policy action.
        """
        return self._exploration_policy.choose_next_action(process_state(state))

    def latest_loss(self) -> float:
        """
        Get the latest loss.

        :return: The latest loss value.
        """
        return self._exploration_policy.latest_loss

    def hyperparameters(self) -> dict:
        """
        Get the hyperparameters of the agent.

        :return: A dictionary containing hyperparameters.
        """
        return self._hyperparameters

    def disable_training(self) -> None:
        """
        Disable training for the agent.
        """
        self._training = False

    def enable_training(self) -> None:
        """
        Enable training for the agent.
        """
        self._training = True