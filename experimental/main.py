from bachelorproject.dqn.dqnagent import DQNAgent
from loops.envinteraction import env_interaction_gym

if __name__ == "__main__":
    agent = DQNAgent("MountainCar-v0", total_frames=5)
    agent.train()
    # env_interaction_gym("gaussian_dp_agent", "MountainCar-v0", 3000)
