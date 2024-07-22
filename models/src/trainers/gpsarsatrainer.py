from typing import Tuple

import torch
from torchrl.data import ReplayBuffer

from bayesopt.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from trainers.trainer import AbstractTrainer


class GPSarsaTrainer(AbstractTrainer):
    """
    Trainer class for SARSA with Gaussian Processes.
    Calculates TD targets for batched trajectories using a Bayesian optimization module.
    """

    def __init__(self,
                 bayesian_opt_module: AbstractBayesianOptimizerRL,
                 batch_size: int,
                 buf: ReplayBuffer,
                 discount_factor: float):
        """
        Constructor for GPSarsaTrainer.

        :param bayesian_opt_module: Bayesian optimization module to use (e.g., a GP to calculate p(q|D)).
        :param batch_size: Number of batches of trajectories used.
        :param buf: A ReplayBuffer from the torchrl library.
        :param discount_factor: `gamma` variable as seen in the literature.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf
        self.bayesian_opt_module = bayesian_opt_module
        self.batch_size = batch_size
        self.discount_factor = discount_factor

    def _trajectory(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, next states, and next actions.
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3], trajectories[4]

    def _construct_target_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of trajectory stored in a buffer, calculate a batch of input-output pairs (x_i, y_i)
        where x_i is a state-action pair and y_i is a temporal difference target.

        :return: state_action_pairs, td_targets.
        """
        states, actions, rewards, next_states, next_actions = self._trajectory()

        if len(actions.shape) == 1:  # Check if shape is [batch_size]
            actions = actions.unsqueeze(1)  # Convert shape to [batch_size, 1]
            next_actions = next_actions.unsqueeze(1)

        state_action_pairs = torch.cat((states, actions), dim=1).to(self.device)

        next_q_values = self.bayesian_opt_module.state_action_value(next_states, next_actions)

        # Compute TD(0) target. Convert rewards from [batch_size] to [batch_size, 1]
        # so that it is compatible with next_q_values of shape [batch_size, 1].
        td_targets = rewards.unsqueeze(1) + self.discount_factor * next_q_values

        return state_action_pairs, td_targets

    def train(self) -> None:
        """
        Performs conditioning on additional training data and optimizes hyperparameters.
        NOTE: This is delegated to AbstractBayesianOptimizerRL.
        """
        state_action_pairs, td_target = self._construct_target_dataset()

        self.bayesian_opt_module.fit(state_action_pairs, td_target, True)