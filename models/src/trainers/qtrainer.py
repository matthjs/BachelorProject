from collections import namedtuple
from threading import current_thread

import torch
from torchrl.data import ReplayBuffer

from trainers.rltrainer import RLTrainer
import torch.nn as nn


class QTrainer(RLTrainer):
    """
    Trainer object for a Q-learning agent. Note, that it assumes one model is used for
    the q-function. Meaning, there is no separate network with frozen parameter like with
    DQN.
    """
    def __init__(self,
                 value_model,
                 target_model,
                 batch_size: int,
                 buf: ReplayBuffer,
                 learning_rate: float = 0.01,
                 discount_factor=0.9,
                 gradient_steps=1):
        """
        NOTE: Expects models to contain a mapping "value_model" -> torch.nn.module
        :param models:
        :param batch_size:
        :param buf:
        :param learning_rate:
        :param discount_factor:
        """
        super().__init__({"value_model": value_model, "target_model": target_model}, batch_size, buf, learning_rate, discount_factor, loss=nn.SmoothL1Loss())
        self.value_model = value_model
        self.target_model = target_model
        self.gradient_steps = gradient_steps
        self.optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def _trajectories(self) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Samples from the memory replay buffer, the reference
        to which is shared with the agent.
        :return: BATCH_SIZE times trajectory tuple (S, A, R, S)
        """
        trajectories = self.buf.sample(batch_size=self.batch_size)

        print("Traj", trajectories)

        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def _compute_trajectory_batches(self) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        state_batch, action_batch, reward_batch, next_state_batch = self._trajectories()

        print("PRE NEXT STATE BATCH", next_state_batch)

        # Compute next_state batch. Here
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_state_batch = torch.cat([s for s in next_state_batch if s is not None])

        return state_batch, action_batch, reward_batch, non_final_next_state_batch, non_final_mask

    def _evaluate_q_network(self, state_batch, action_batch):
        """
        Compute q values of neural network approximating
        the action-value function for the chosen actions
        in the batch.
        :return: batched model output q(S,A) for all actions.
        """
        model_output = self.value_model(state_batch)
        q_values = []
        print("input shape ->", state_batch.shape)
        print(state_batch)
        for idx, vec in enumerate(model_output):
            # print("idx, vec", idx, vec)
            q_values.append(vec[action_batch[idx]])

        return torch.as_tensor(q_values, device=self.device, dtype=torch.float64)

    def evaluate_target_network(self, next_state_batch, reward_batch, mask):
        """
        The computation here takes into account whether we are in a final state or not.
        Because if next_state is a final state then the target is just the reward.
        Based on stable-baselines DQN implementation.
        :param next_state_batch:
        :param mask:
        :return: the TD target.
        """
        # DQN would use torch.no_grad, but I guess in this case you will not.

        with torch.no_grad():
            # max_q_value, _ = torch.max(self.target_model(next_state_batch), dim=0)
            print("Target input shape ->", next_state_batch.shape)
            print("Target input ->", next_state_batch)

            next_state_values = self.target_model(next_state_batch).max()
            print("target output shape ->", next_state_values.shape)
            print("Target output ->", next_state_values)

            td_target = (next_state_values * self.discount_factor) + reward_batch
        return td_target

    def train(self) -> None:
        """
        Perform the gradient optimization step on the neural network.
        Based on -> tps://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        if len(self.buf) < self.batch_size:
            return

        for _ in range(self.gradient_steps):
            state_batch, action_batch, reward_batch, next_state_batch, non_final_mask = self._compute_trajectory_batches()

            state_action_values = self._evaluate_q_network(state_batch, action_batch)
            td_target = self.evaluate_target_network(next_state_batch, reward_batch, non_final_mask)

            # The loss is the MSE of the q-output of the current state and the temporal difference target.
            # Note that the TD target is off-policy in Q-learning.
            loss = self.loss_fn(state_action_values, td_target)

            # gradient descent step
            self.optimizer.zero_grad()
            loss.required_grad = True
            loss.backward()
            # torch.nn.utils.clip_grad_value_(self.value_model.parameters(), 100)
            self.optimizer.step()

            # MetricsTracker().record_loss(current_thread().name, loss.item())

