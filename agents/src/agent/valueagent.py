from abc import ABC

from torchrl.data import ReplayBuffer

from agent.agent import Agent
from trainers.rltrainer import RLTrainer


class ValueAgent(Agent, ABC):
    def __init__(self, models, memory: ReplayBuffer, exploration_policy, trainer: RLTrainer):
        super().__init__(models, memory)
        self._exploration_policy = exploration_policy
        self._trainer = trainer

    def update(self) -> None:
        """
        Update Q-estimate and exploration (epsilon) factor.
        Epsilon only updated upon environment end.
        :return:
        """
        if self.env_info.done:
            self._exploration_policy.step()  # Adjust epsilon
        self._trainer.train()  # Gradient update step ~ TD target.

    def policy(self, state):
        """
        Delegates action selection to the epsilon greedy class.
        :param state: of the environment.
        :return: an action.
        """
        return self._exploration_policy.action(state)
