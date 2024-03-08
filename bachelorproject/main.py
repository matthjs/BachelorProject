from stable_baselines3.dqn import MlpPolicy

from agentfactory.agentfactory import AgentFactory
from loops.envinteraction import env_interaction_gym
from stable_baselines3 import DQN
import gymnasium as gym

from metricstracker.metricstracker import MetricsTracker

if __name__ == "__main__":
    # env_interaction_gym("mlp_q_agent", "MountainCar-v0", 3000)
    agent = AgentFactory().create_agent("gaussian_dp_agent", gym.make("MountainCar-v0", render_mode="human"))
    agent.update()
