from collections import namedtuple
from threading import current_thread

import torch
from torchrl.data import ReplayBuffer

from metricstracker.metricstracker import MetricsTracker
from trainers.rltrainer import RLTrainer
import torch.nn as nn


class QTrainer(RLTrainer):
    def __init__(self, model: nn.Module, batch_size: int,
                 buf: ReplayBuffer, learning_rate: float = 0.001, discount_factor=0.9):
        super().__init__(model, batch_size, buf, learning_rate, discount_factor, loss=nn.MSELoss())

    def _trajectories(self) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Samples from the memory replay buffer, the reference
        to which is shared with the agent.
        :return: BATCH_SIZE times trajectory tuple (S, A, R, S)
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)

        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def _compute_trajectory_batches(self) -> tuple:
        state_batch, action_batch, reward_batch, next_state_batch = self._trajectories()

        # TODO: determine if the following is necessary.
        # Compute next_state batch.
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)),
        #                              device=self.device, dtype=torch.bool)
        #next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        return state_batch, action_batch, reward_batch, next_state_batch

    def _evaluate_q_network(self, state_batch, action_batch):
        """
        Compute q values of neural network approximating
        the action-value function for the chosen actions
        in the batch.
        Note that action selection here is off-policy (due to max operation).
        :return:
        """
        model_output = self.model(state_batch)
        q_values = []
        for idx, vec in enumerate(model_output):
            q_values.append(vec[action_batch[idx]])

        return torch.as_tensor(q_values, device=self.device, dtype=torch.float64)

    def _evaluate_target_network(self, next_state_batch, reward_batch, mask):
        """
        :param next_state_batch:
        :param mask:
        :return: the TD target.
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # DQN would use torch.no_grad but I guess in this case you will not have this.
        next_state_values[mask] = self.model(next_state_batch).max(1)[0]

        td_target = (next_state_values * self.discount_factor) + reward_batch
        return td_target

    def train(self) -> None:
        """
        Perform the gradient optimization step on the neural network.
        Based on -> tps://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        if len(self.buf) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask = self._compute_trajectory_batches()

        state_action_values = self._evaluate_q_network(state_batch, action_batch)
        td_target = self._evaluate_target_network(next_state_batch, reward_batch, non_final_mask)

        loss = self.loss_fn(state_action_values, td_target)

        # gradient descent step
        self.optimizer.zero_grad()
        loss.required_grad = True
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        MetricsTracker().record_loss(current_thread().name, loss.item())

