from stable_baselines3.common.base_class import BaseAlgorithm

from agent.abstractagent import AbstractAgent


class StableBaselinesAdapter(AbstractAgent):
    def __init__(self, model: BaseAlgorithm):
        super().__init__({}, model.observation_space, model.action_space)
        self._sb_model = model

    def add_trajectory(self, trajectory: tuple) -> None:
        pass

    def update(self):
        pass

    def policy(self, state):
        return self._sb_model.predict(state)[0]

    def is_stable_baselines_wrapper(self):
        return True

    def stable_baselines_unwrapped(self):
        return self._sb_model
