from loops.envinteraction import env_interaction_gym
from loops.stablebaselinesrun import dqn_train, dqn_evaluate_policy, dqn_train2, dqn_play_cartpole

if __name__ == "__main__":
    env_interaction_gym("gpq_agent", "CartPole-v1", 3000)
    # dqn_train2()
    # dqn_play_cartpole()
    # dqn_evaluate_policy()
