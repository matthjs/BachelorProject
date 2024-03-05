from agent.abstractagent import AbstractAgent
from agent.dqnagent import DQNAgent
from builders.valueagentbuilder import ValueAgentBuilder


class DQNAgentBuilder(ValueAgentBuilder):
    def __init__(self):
        super().__init__()
        self.param_copying = None
        self.train_count = 0

    def set_param_copying(self, param_copying: int):
        self.param_copying = param_copying
        return self

    def valid(self) -> bool:
        return super().valid() and self.param_copying is not None

    def build(self) -> AbstractAgent:
        if self.valid():
            return DQNAgent(self)
        else:
            raise ValueError("Not all required attributes are set.")
