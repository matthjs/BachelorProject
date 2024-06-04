from agent.valueagent import ValueAgent
from exploration.egreedy import EpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from trainers.qtrainer import QTrainer


# ! Not used !


class QAgent(ValueAgent):
    def __init__(self,
                 models,
                 state_space,
                 action_space,
                 replay_buffer_size: int,
                 value_model_type_str,
                 batch_size=1,
                 annealing_num_steps=2000,
                 learning_rate=0.01,
                 discount_factor=0.9):
        # The 'Product' is also the 'Director'
        super().__init__(models, state_space, action_space, replay_buffer_size)
        self._models["value_model"] = ModelFactory.create_model(value_model_type_str,
                                                                state_space.shape[0],
                                                                action_space.n)
        self._exploration_policy = EpsilonGreedy(self._models["value_model"],
                                                 action_space,
                                                 annealing_num_steps=annealing_num_steps)
        self._trainer = QTrainer(self._models["value_model"],
                                 self._models["value_model"],
                                 batch_size=batch_size,
                                 buf=self._replay_buffer,
                                 learning_rate=learning_rate,
                                 discount_factor=discount_factor)

        print(f"CONSTRUCTED Q AGENT WITH BATCH_SIZE:{batch_size}, ANNEALING_NUM_STEPS:{annealing_num_steps},"
              f"learning_rate:{learning_rate}, DISCOUNT_FACTOR:{discount_factor}")

        self._hyperparameters = {
            'value_model': value_model_type_str,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            'annealing_num_steps': annealing_num_steps,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor
        }

    def hyperparameters(self) -> dict:
        return self._hyperparameters
