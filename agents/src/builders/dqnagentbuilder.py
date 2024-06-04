from agent.abstractagent import AbstractAgent
from agent.dqnagent import DQNAgent
from builders.valueagentbuilder import ValueAgentBuilder


# ! Not used !


class DQNAgentBuilder(ValueAgentBuilder):
    def __init__(self):
        super().__init__()
        self.param_copying = None

    def set_param_copying(self, param_copying: int):
        self.param_copying = param_copying
        return self

    def build(self) -> AbstractAgent:
        if self.valid():
            return DQNAgent(
                models=self.models,
                state_space=self.state_space,
                action_space=self.action_space,
                replay_buffer_size=self.replay_buffer_size,
                value_model_type_str=self.value_model_type_str,
                batch_size=self.batch_size,
                annealing_num_steps=self.annealing_num_steps,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                param_copying=self.param_copying
            )
        else:
            raise ValueError("Not all required attributes are set.")
