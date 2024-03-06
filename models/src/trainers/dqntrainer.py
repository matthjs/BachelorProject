import torch
from overrides import override
from torchrl.data import ReplayBuffer

import torch.nn as nn

from trainers.qtrainer import QTrainer
from trainers.rltrainer import RLTrainer


class DQNTrainer(QTrainer):
    def __init__(self,
                 models,
                 batch_size: int,
                 buf: ReplayBuffer,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.9):
        super().__init__(models, batch_size, buf, learning_rate, discount_factor)
        self.target_model = self.models["target_model"]

    @override
    def _evaluate_target_network(self, next_state_batch, reward_batch, mask):
        """
        The computation here takes into account whether we are in a final state or not.
        Because if next_state is a final state then the target is just the reward.
        :param next_state_batch:
        :param mask:
        :return: the TD target.
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # DQN would use torch.no_grad, but I guess in this case you will not.

        with torch.no_grad():
            next_state_values[mask] = self.target_model(next_state_batch).max()

        td_target = (next_state_values * self.discount_factor) + reward_batch
        return td_target
