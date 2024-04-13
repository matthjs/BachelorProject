from torchrl.data import ReplayBuffer, TensorDictReplayBuffer

from agent.abstractagent import AbstractAgent
from exploration.gpepsilongreedy import GPEpsilonGreedy
from modelfactory.modelfactory import ModelFactory
from trainers.gptrainer import GaussianProcessTrainer


class GPQAgent(AbstractAgent):
    def __int__(self,
                gp_model_str: str,
                state_space,
                action_space,
                learning_rate: float = 0.01,
                discount_factor: float = 0.99,
                annealing_num_steps=2000,
                batch_size=1,
                sparsification=False):
        self._models["value_model"] = ModelFactory.create_model(gp_model_str,
                                                                state_space.shape[0],
                                                                action_space.n)

        self._exploration_policy = GPEpsilonGreedy(self._models["value_model"],
                                                   action_space,
                                                   annealing_num_steps=annealing_num_steps)
        self._trainer = GaussianProcessTrainer(self._models["value_model"],
                                               learning_rate=learning_rate)
        pass

    def update(self):
        pass

    def policy(self, state):
        return self._exploration_policy.action(state)
