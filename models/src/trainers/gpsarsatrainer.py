from typing import Tuple

import torch
from torchrl.data import ReplayBuffer

from models.gp import ExactGaussianProcessRegressor, max_state_action_value
from trainers.gptrainer import GaussianProcessTrainer
from trainers.trainer import AbstractTrainer


class ExactGPSarsaTrainer(AbstractTrainer):
    def __init__(self,
                 model: ExactGaussianProcessRegressor,
                 action_space_size: int,  # Assumes actions are encoded as 0, 1, 2, ..., action_space_size
                 batch_size: int,
                 buf: ReplayBuffer,
                 learning_rate: float,
                 discount_factor: float,
                 num_epochs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf
        self.action_space_size = action_space_size

        self.gp = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.gp_trainer = GaussianProcessTrainer(
            self.gp,
            learning_rate=learning_rate,
        )

        self.num_epochs = num_epochs

    def _trajectory(self) -> tuple:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3], trajectories[4]

    def _construct_target_dataset(self) -> Tuple[torch.tensor, torch.tensor]:
        states, actions, rewards, next_states, next_actions = self._trajectory()

        # TODO: see if this can be more flexible.
        if len(actions.shape) == 1:  # Check if shape is [batch_size]
            actions = actions.unsqueeze(1)  # Convert shape to [batch_size, 1]

        print("action: ", actions.shape)

        state_action_pairs = torch.cat((states, actions), dim=1).to(self.device)
        next_state_action_pairs = torch.cat((next_states, next_actions), dim=1).to(self.device)

        next_q_values, _, _, _ = self.gp.predict(next_state_action_pairs)
        # Compute TD(0) target.

        # Convert rewards from [32] -> [32, 1] so that it is compatible with max_q_values of shape [32, 1]
        td_targets = rewards.unsqueeze(1) + self.discount_factor * next_q_values

        print("TD->", td_targets)

        return state_action_pairs, td_targets

    def train(self) -> None:
        """
        Performs conditioning on additional training data + optionally refits kernel hyperparameters.
        NOTE: This is delegated to GaussianProcessTrainer.
        """
        state_action_pairs, td_target = self._construct_target_dataset()

        # NOTE: performs MLL estimation of kernel parameters every time.
        self.gp_trainer.train(state_action_pairs, td_target, 100, True, True)