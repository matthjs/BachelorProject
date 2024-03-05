from abc import ABC

from agent.abstractagent import AbstractAgent
from builders.valueagentbuilder import ValueAgentBuilder


class ValueAgent(AbstractAgent, ABC):

    def __init__(self, builder: ValueAgentBuilder):
        """
        NOTE: This constructor should not be run by itself but indirectly by the builder.
        :param builder:
        """
        super().__init__(builder)
        self._exploration_policy = builder.exploration_policy()
        self._trainer = builder.trainer()

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
