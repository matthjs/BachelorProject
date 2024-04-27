from typing import Tuple

import torch
from torchrl.data import ReplayBuffer

from models.gp import ExactGaussianProcessRegressor
from trainers.gptrainer import GaussianProcessTrainer
from trainers.trainer import AbstractTrainer


class ExactGPQTrainer(AbstractTrainer):
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
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def construct_target_dataset(self) -> Tuple[torch.tensor, torch.tensor]:
        states, actions, rewards, next_states = self._trajectory()
        print("state: ", states.shape)

        # TODO: see if this can be more flexible.
        if len(actions.shape) == 1:  # Check if shape is [batch_size]
            actions = actions.unsqueeze(1)  # Convert shape to [batch_size, 1]

        print("action: ", actions.shape)

        state_action_pairs = torch.cat((states, actions), dim=1).to(self.device)

        # Compute TD(0) target.
        # The max() operation over actions inherently means this can only be done
        # for discrete action spaces.
        with torch.no_grad():
            q_values = []

            # Again assume discrete action encoding starting from 0.
            for action in range(self.action_space_size):
                action_batch = torch.full((self.batch_size, 1), action).to(self.device)
                next_state_action_pair = torch.cat((states, action_batch), dim=1).to(self.device)
                mean_q, _, _, _ = self.gp.predict(next_state_action_pair)
                q_values.append(mean_q)

            # Some reshaping black magic to get the max q value along each batch dimension.
            q_tensor = torch.cat(q_values, dim=0).view(len(q_values), -1, 1)
            max_q_values, _ = torch.max(q_tensor, dim=0)

        td_targets = rewards + self.discount_factor * max_q_values

        return state_action_pairs, td_targets

    def train(self) -> None:
        state_action_pairs, td_target = self.construct_target_dataset()

        print("YOOOO!")

        # NOTE: performs MLL estimation of kernel parameters every time.
        self.gp_trainer.train(state_action_pairs, td_target)
