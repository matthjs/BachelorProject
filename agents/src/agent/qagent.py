from agent.valueagent import ValueAgent
from builders.qagentbuilder import QAgentBuilder


class QAgent(ValueAgent):
    def __init__(self, builder: QAgentBuilder):
        super().__init__(builder)
