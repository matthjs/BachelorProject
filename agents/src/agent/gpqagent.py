import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement

from agent.abstractagent import AbstractAgent
from exploration.gpepsilongreedy import GPEpsilonGreedy
from gp.bayesianoptimizer_rl import BayesianOptimizerRL
from modelfactory.modelfactory import ModelFactory
from trainers.gpqtrainer import GPQTrainer
from trainers.gptrainer import GaussianProcessTrainer
from util.fetchdevice import fetch_device
from util.processstate import process_state


class GPQAgent(AbstractAgent):
    def __init__(self,
                 gp_model_str: str,
                 state_space,
                 action_space,
                 discount_factor: float,
                 batch_size,
                 replay_buffer_size,
                 exploring_starts,
                 max_dataset_size,
                 sparsification=False):
        super(GPQAgent, self).__init__({}, state_space, action_space)

        self._exploration_policy = BayesianOptimizerRL(
            model_str=gp_model_str,
            max_dataset_size=max_dataset_size,
            random_draws=exploring_starts,
            state_size=state_space.shape[0],
            action_space=action_space,
        )

        self._replay_buffer = ReplayBuffer(storage=LazyTensorStorage(
                                           max_size=replay_buffer_size,
                                           device=fetch_device()),
                                           sampler=SamplerWithoutReplacement())

        self._batch_counter = 0
        self._batch_size = batch_size

        self._trainer = GPQTrainer(
            self._exploration_policy,
            batch_size=batch_size,
            buf=self._replay_buffer,
            discount_factor=discount_factor
        )

    def update(self):
        if self._batch_counter >= self._batch_size:
            self._trainer.train()
            self._batch_counter = 0
        self._batch_counter += 1

    # noinspection DuplicatedCode
    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the replay buffer.
        NOTE: trajectory is converted to tensor and moved to self.device.
        :param trajectory = (state, action, reward, next_state)
        """
        state, action, reward, next_state = trajectory
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.double)
        action_t = torch.as_tensor(action, device=self.device)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.double)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.double)
        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t))

    def policy(self, state):
        return self._exploration_policy.choose_next_action(process_state(state))
