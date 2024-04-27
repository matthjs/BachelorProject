import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.abstractagent import AbstractAgent
from exploration.gpepsilongreedy import GPEpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from trainers.gpqtrainer import ExactGPQTrainer
from trainers.gptrainer import GaussianProcessTrainer
from util.fetchdevice import fetch_device


class GPQAgent(AbstractAgent):
    def __init__(self,
                 gp_model_str: str,
                 state_space,
                 action_space,
                 learning_rate: float,
                 discount_factor: float,
                 annealing_num_steps,
                 batch_size,
                 replay_buffer_size,
                 num_epochs,
                 sparsification=False):
        super(GPQAgent, self).__init__({}, state_space, action_space)
        self._models["value_model"] = ModelFactory.create_model(gp_model_str,
                                                                state_space.shape[0],
                                                                action_space.n)

        self._exploration_policy = GPEpsilonGreedy(model=self._models["value_model"],
                                                   action_space=action_space,
                                                   annealing_num_steps=annealing_num_steps)
        self._replay_buffer = ReplayBuffer(storage=LazyTensorStorage(
                                           max_size=replay_buffer_size,
                                           device=fetch_device()))

        if gp_model_str == "exact_gp":
            self._trainer = ExactGPQTrainer(
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
        self._trainer.train()

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
        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t))

    def policy(self, state):
        return self._exploration_policy.action(state)
