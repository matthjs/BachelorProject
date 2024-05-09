from agent.abstractagent import AbstractAgent
import gymnasium as gym


class RandomAgent(AbstractAgent):
    def __init__(self, env: gym.Env):
        super().__init__({}, env.observation_space, env.action_space)
        self.action_space = env.action_space

    def add_trajectory(self, trajectory):
        pass

    def update(self) -> None:
        pass

    def policy(self, state):
        return self.action_space.sample()
