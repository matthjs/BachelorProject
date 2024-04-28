from loops.envinteraction import env_interaction_gym
from loops.stablebaselinesrun import dqn_train, dqn_play_mountain_car

if __name__ == "__main__":
    # env_interaction_gym("gpq_agent", "MountainCar-v0", 3000)
    dqn_train()
    dqn_play_mountain_car()
