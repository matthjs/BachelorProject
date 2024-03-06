from agent.valueagent import ValueAgent
from builders.qagentbuilder import QAgentBuilder
from exploration.egreedy import EpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from trainers.qtrainer import QTrainer


class QAgent(ValueAgent):
    def __init__(self, builder: QAgentBuilder):
        super().__init__(builder)
        self._exploration_policy = builder.exploration_policy
        self._trainer = builder.trainer

        # The 'Product' is also the 'Director'
        self.models["value_model"] = ModelFactory.create_model(builder.value_model_type_str,
                                                               builder.state_space.shape[0],
                                                               builder.action_space.n)
        self.exploration_policy = EpsilonGreedy(self.models["value_model"],
                                                builder.action_space,
                                                annealing_num_steps=builder.annealing_num_steps)
        self.trainer = QTrainer(self.models,
                                batch_size=builder.batch_size,
                                buf=self._replay_buffer,
                                learning_rate=builder.learning_rate,
                                discount_factor=builder.discount_factor)