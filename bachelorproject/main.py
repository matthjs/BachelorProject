from stable_baselines3.dqn import MlpPolicy

from agentfactory.agentfactory import AgentFactory
import gymnasium as gym

from loops.envinteraction import env_interaction_gym
from metricstracker.metricstracker import MetricsTracker

if __name__ == "__main__":
    env_interaction_gym("gaussian_dp_agent", "MountainCar-v0", 3000)
