from agent.abstractagent import AbstractAgent
from agent.abstractdpagent import AbstractDPAgent
import gymnasium as gym


class GaussianProcessDPAgent(AbstractDPAgent):
    """
    Dynamic Programming Agent.
    """

    def __init__(self, models, env: gym.Env):
        super().__init__(models, env.observation_space, env.action_space)
        self.policy_calculated = False

    def iterate(self):
        self.policy_calculated = True

    def policy(self, state):
        if self.policy_calculated is False:
            print("Optimal Policy Not calculated yet, running policy iteration")
            self.iterate()
