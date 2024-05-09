from typing import Tuple

import torch
from torchrl.data import ReplayBuffer

from gp.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from trainers.gptrainer import GaussianProcessTrainer
from trainers.trainer import AbstractTrainer


class GPQTrainer(AbstractTrainer):
    def __init__(self,
                 bayesian_opt_module: AbstractBayesianOptimizerRL,
                 batch_size: int,
                 buf: ReplayBuffer,
                 discount_factor: float):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf

        self.bayesian_opt_module = bayesian_opt_module
        self.batch_size = batch_size
        self.discount_factor = discount_factor

    def _trajectory(self) -> tuple:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def _construct_target_dataset(self) -> Tuple[torch.tensor, torch.tensor]:
        # noinspection DuplicatedCode
        states, actions, rewards, next_states = self._trajectory()
        # print("state: ", states.shape)

        # TODO: see if this can be more flexible.
        if len(actions.shape) == 1:  # Check if shape is [batch_size]
            actions = actions.unsqueeze(1)  # Convert shape to [batch_size, 1]

        # print("action: ", actions.shape)

        state_action_pairs = torch.cat((states, actions), dim=1).to(self.device)

        # covar_test(self.gp, state_action_pairs)

        # Compute TD(0) target.
        # The max() operation over actions inherently means this can only be done
        # for discrete action spaces.
        max_q_values, _ = self.bayesian_opt_module.max_state_action_value(next_states)

        # print("max_q -> ", max_q_values)

        # Convert rewards from [32] -> [32, 1] so that it is compatible with max_q_values of shape [32, 1]
        td_targets = rewards.unsqueeze(1) + self.discount_factor * max_q_values

        # print("TD->", td_targets)

        return state_action_pairs, td_targets

    def train(self) -> None:
        """
        Performs conditioning on additional training data + optionally refits kernel hyperparameters.
        NOTE: This is delegated to GaussianProcessTrainer.
        """
        state_action_pairs, td_target = self._construct_target_dataset()

        # NOTE: performs MLL estimation of kernel parameters every time.
        self.bayesian_opt_module.fit(state_action_pairs, td_target, True)
