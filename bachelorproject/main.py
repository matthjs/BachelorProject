import torch

from gp.bayesianoptimizer_rl import append_actions
from loops.envinteraction import env_interaction_gym, env_interaction_gym2
from loops.stablebaselinesrun import dqn_train, dqn_evaluate_policy, dqn_train2, dqn_play_cartpole

if __name__ == "__main__":
    # state = torch.tensor([[1, 2]])  # Replace with your state vector
    # action_size = 5
    # res = append_actions(state, action_size)
    # print(res)
    # print(res.shape)

    env_interaction_gym2("random", "CartPole-v1", 100)
    env_interaction_gym2("gpq_agent", "CartPole-v1", 100)
    env_interaction_gym2("gpsarsa_agent", "CartPole-v1", 100)
    # dqn_train2()
    # dqn_play_cartpole()
    # dqn_evaluate_policy()
