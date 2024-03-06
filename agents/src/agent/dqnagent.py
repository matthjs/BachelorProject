from agent.valueagent import ValueAgent
from exploration.egreedy import EpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from trainers.dqntrainer import DQNTrainer


class DQNAgent(ValueAgent):
    def __init__(self,
                 models,
                 state_space,
                 action_space,
                 replay_buffer_size: int,
                 value_model_type_str,
                 batch_size=64,
                 annealing_num_steps=2000,
                 learning_rate=0.01,
                 discount_factor=0.9,
                 param_copying=20):
        # The 'Product' is also the 'Director'
        super().__init__(models, state_space, action_space, replay_buffer_size)
        self._models["value_model"] = ModelFactory.create_model(value_model_type_str,
                                                                state_space.shape[0],
                                                                action_space.n)
        self._models["target_model"] = ModelFactory.create_model(value_model_type_str,
                                                                 state_space.shape[0],
                                                                 action_space.n)
        self._exploration_policy = EpsilonGreedy(self.models["value_model"],
                                                 action_space,
                                                 annealing_num_steps=annealing_num_steps)
        self._trainer = DQNTrainer(self.models,
                                   batch_size=batch_size,
                                   buf=self._replay_buffer,
                                   learning_rate=learning_rate,
                                   discount_factor=discount_factor)
        self._train_count = 0
        self._param_copying = param_copying

    def update(self) -> None:
        if self._train_count == self._param_copying:
            self._train_count = 0
            self.models["target_network"].load_state_dict(self.models["value_network"].state_dict())

        super().update()
        self._train_count += 1
