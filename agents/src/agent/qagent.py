import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage

from agent.agent import Agent
from util.fetchdevice import fetch_device
from trainers.qtrainer import QTrainer


class QAgent(Agent):
    def __init__(self, model, exploration_policy):
        """
        Simple implementation of a Q-learning agent with function approximation
        for Q function.
        :param model: for function approximation.
        :param exploration_policy: i.e., epsilon greedy.
        """
        # Note: Classical Q-learning does not use a ReplayBuffer, so we set
        # the ReplayBuffer size to 1.
        super().__init__({"value_network": model},
                         memory=ReplayBuffer(storage=LazyTensorStorage(1, device=fetch_device())))
        self._exploration_policy = exploration_policy
        self._trainer = QTrainer(model=model, batch_size=1, buf=self._replay_buffer)

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a to replay buffer.
        NOTE: trajectory is converted to tensor and moved to self.device.
        :param trajectory = (state, action, reward, next_state)
        """
        state, action, reward, next_state = trajectory
        state_t = torch.as_tensor(state, device=self.device)
        action_t = torch.as_tensor(action, device=self.device)
        reward_t = torch.as_tensor(reward, device=self.device)
        next_state_t = torch.as_tensor(next_state, device=self.device)
        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t))

    def update(self) -> None:
        """
        Update Q-estimate and exploration (epsilon) factor.
        Epsilon only updated upon environment end.
        :return:
        """
        if self.env_info.done:
            self._exploration_policy.step()    # Adjust epsilon
        self._trainer.train()              # Gradient update step ~ TD target.

    def policy(self, state):
        """
        Delegates action selection to the epsilon greedy class.
        :param state: of the environment.
        :return: an action.
        """
        return self._exploration_policy.action(state)

    def load_parameters(self):
        for model_name, model in self.models:
            model.load()

    def save_parameters(self):
        for model_name, model in self.models:
            model.save()
