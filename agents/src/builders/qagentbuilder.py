from agent.abstractagent import AbstractAgent
from agent.qagent import QAgent
from builders.valueagentbuilder import ValueAgentBuilder


# ! Not used !


class QAgentBuilder(ValueAgentBuilder):
    def __init__(self):
        super().__init__()

    def build(self) -> AbstractAgent:
        if self.valid():
            return QAgent(
                models=self.models,
                state_space=self.state_space,
                action_space=self.action_space,
                replay_buffer_size=self.replay_buffer_size,
                value_model_type_str=self.value_model_type_str,
                batch_size=self.batch_size,
                annealing_num_steps=self.annealing_num_steps,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor
            )
        else:
            raise ValueError("Not all required attributes are set.")
