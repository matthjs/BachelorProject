from typing import Tuple

import torch
from torchrl.data import ReplayBuffer

from bayesopt.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from trainers.trainer import AbstractTrainer


class GPQTrainer(AbstractTrainer):
    """
    Calculates TD targets for batched trajectories. Uses a Bayesian optimization module
    to perform bayesian update.
    """

    def __init__(self,
                 bayesian_opt_module: AbstractBayesianOptimizerRL,
                 batch_size: int,
                 buf: ReplayBuffer,
                 discount_factor: float):
        """
        Constructor for GPQTrainer.

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

    def _trajectory(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def _construct_target_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of trajectory stored in a buffer, calculate a batch of input-output pairs (x_i, y_i)
        where x_i is a state-action pair and y_i is a temporal difference target.

        :return: state_action_pairs, td_targets.
        """
        # noinspection DuplicatedCode
        states, actions, rewards, next_states = self._trajectory()

        if len(actions.shape) == 1:  # Check if shape is [batch_size]
            actions = actions.unsqueeze(1)  # Convert shape to [batch_size, 1]

        state_action_pairs = torch.cat((states, actions), dim=1).to(self.device)

        # Compute TD(0) target.
        max_q_values, _ = self.bayesian_opt_module.max_state_action_value(next_states)
        # Convert rewards from [batch_size] -> [batch_size, 1] so that it is compatible with max_q_values of shape [
        # batch_size, 1]

        # Compute TD(0) target.
        td_targets = rewards.unsqueeze(1) + self.discount_factor * max_q_values

        return state_action_pairs, td_targets

    def train(self) -> None:
        """
        Performs conditioning on additional training data and optimizes hyperparameters.
        NOTE: This is delegated to AbstractBayesianOptimizerRL.
        """
        state_action_pairs, td_target = self._construct_target_dataset()

        self.bayesian_opt_module.fit(state_action_pairs, td_target, True)
