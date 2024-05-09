import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement

from agent.abstractagent import AbstractAgent
from exploration.gpepsilongreedy import GPEpsilonGreedy
from gp.bayesianoptimizer_rl import BayesianOptimizerRL
from modelfactory.modelfactory import ModelFactory
from trainers.gpqtrainer import GPQTrainer
from trainers.gpsarsatrainer import GPSarsaTrainer
from util.fetchdevice import fetch_device


class GPSarsaAgent(AbstractAgent):
    """
    One advantage of the SARSA agent is that it is actually compatible with continuous action spaces
    provided you do not have greedy action selection.
    """
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
        super(GPSarsaAgent, self).__init__({}, state_space, action_space)

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
                                           sampler=SamplerWithoutReplacement())    # IMPORTANT FOR ON-POLICY.

        self._batch_counter = 0
        self._batch_size = batch_size

        if gp_model_str == "exact_gp":
            self._trainer = GPSarsaTrainer(
                model=self._models["value_model"],
                action_space_size=action_space.n,
                batch_size=batch_size,
                buf=self._replay_buffer,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                num_epochs=num_epochs
            )
        else:
            raise ValueError(f"No trainer for gaussian process model `{gp_model_str}`")

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
        state_t = torch.as_tensor(state, device=self.device)
        action_t = torch.as_tensor(action, device=self.device)
        reward_t = torch.as_tensor(reward, device=self.device)
        next_state_t = torch.as_tensor(next_state, device=self.device)

        # To ensure compatible interface with on-policy algorithms we have to run the policy one more time here.
        next_action = self.policy(next_state)
        next_action_t = torch.as_tensor(next_action, device=self.device)

        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t, next_action_t))

    def policy(self, state):
        return self._exploration_policy.action(state)