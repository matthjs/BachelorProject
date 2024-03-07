from stable_baselines3.dqn import MlpPolicy

from loops.envinteraction import env_interaction_gym
from stable_baselines3 import DQN
import gymnasium as gym

from metricstracker.metricstracker import MetricsTracker

if __name__ == "__main__":
    env_interaction_gym("mlp_q_agent", "MountainCar-v0", 3000)