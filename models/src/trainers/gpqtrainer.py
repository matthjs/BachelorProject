from typing import Tuple

import torch
from torchrl.data import ReplayBuffer

from bayesopt.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from trainers.trainer import AbstractTrainer


class GPQTrainer(AbstractTrainer):
    """
    Calculates TD targets for batched trajectories. Uses a bayesian optimization module
    to perform bayesian update.
    """
    def __init__(self,
                 bayesian_opt_module: AbstractBayesianOptimizerRL,
                 batch_size: int,
                 buf: ReplayBuffer,
                 discount_factor: float):
        """
        Constructor.
        :param bayesian_opt_module: bayesian optimization module to use (e.g., a GP to calculate p(q|D))
        :param batch_size: how many batches of trajectories are used.
        :param buf: a ReplayBuffer from the torchrl library.
        :param discount_factor: `gamma` variable as is seen in the literature.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf

        self.bayesian_opt_module = bayesian_opt_module
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = 0.01 # .0023

    def _trajectory(self) -> tuple:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def _construct_target_dataset(self) -> Tuple[torch.tensor, torch.tensor]:
        """
        Given a batch of trajectory stored in a buffer, calculate a batch of input output pairs (x_i, y_i)
        where x_i is a state action pair and y_i is a temporal difference target.
        :return: state_action_pairs, td_targets.
        """
        # noinspection DuplicatedCode
        states, actions, rewards, next_states = self._trajectory()

        # TODO: see if this can be more flexible.
        if len(actions.shape) == 1:  # Check if shape is [batch_size]
            actions = actions.unsqueeze(1)  # Convert shape to [batch_size, 1]

        state_action_pairs = torch.cat((states, actions), dim=1).to(self.device)

        # Compute TD(0) target.
        # The max() operation over actions inherently means this can only be done
        # for discrete action spaces.
        current_q_values = self.bayesian_opt_module.state_action_value(states, actions)
        max_q_values, _ = self.bayesian_opt_module.max_state_action_value(next_states)
        # Convert rewards from [batch_size] -> [batch_size, 1] so that it is compatible with max_q_values of shape [
        # batch_size, 1]
        # print(current_q_values)
        # td_targets = rewards.unsqueeze(1) + self.discount_factor * max_q_values
        td_targets = current_q_values + self.learning_rate * (
                    rewards.unsqueeze(1) + self.discount_factor * max_q_values - current_q_values)

        return state_action_pairs, td_targets

    def train(self) -> None:
        """
        Performs conditioning on additional training data + optionally refits kernel hyperparameters.
        NOTE: This is delegated to AbstractBayesianOptimizerRL.
        """
        state_action_pairs, td_target = self._construct_target_dataset()

        # NOTE: performs MLL estimation of kernel parameters every time.
        self.bayesian_opt_module.fit(state_action_pairs, td_target, True)
