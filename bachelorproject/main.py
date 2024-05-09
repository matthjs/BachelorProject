import torch

from gp.bayesianoptimizer_rl import append_actions
from loops.envinteraction import env_interaction_gym
from loops.stablebaselinesrun import dqn_train, dqn_evaluate_policy, dqn_train2, dqn_play_cartpole

if __name__ == "__main__":
    # state = torch.tensor([[1, 2]])  # Replace with your state vector
    # action_size = 5
    # res = append_actions(state, action_size)
    # print(res)
    # print(res.shape)

    env_interaction_gym("gpsarsa_agent", "CartPole-v1", 3000)
    env_interaction_gym("random", "CartPole-v1", 3000)
    # dqn_train2()
    # dqn_play_cartpole()
    # dqn_evaluate_policy()
