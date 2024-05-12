import torch

from gp.bayesianoptimizer_rl import append_actions
from loops.envinteraction import env_interaction_gym, env_interaction_gym2
from loops.stablebaselinesrun import dqn_train, dqn_evaluate_policy, dqn_train2, dqn_play_cartpole
from simulators.simulator_rl import SimulatorRL

if __name__ == "__main__":
    sim = SimulatorRL("CartPole-v1")
    (sim.register_agent("gpq_agent_1", "gpq_agent", None)
     .register_agent("gpsarsa_agent_1", "gpsarsa_agent", None)
     .train_agents(num_episodes=100, concurrent=False, logging=True)
     .evaluate_agents("CartPole-v1", 20, None, False, True)
     .data_to_csv()
     .plot_any_plottable_data())

    sim.play("gpq_agent1", 10)

    # state = torch.tensor([[1, 2]])  # Replace with your state vector
    # action_size = 5
    # res = append_actions(state, action_size)
    # print(res)
    # print(res.shape)

    # env_interaction_gym2("random", "CartPole-v1", 100)
    # env_interaction_gym2("gpq_agent", "CartPole-v1", 100)
    # env_interaction_gym2("gpsarsa_agent", "CartPole-v1", 100)
    # dqn_train2()
    # dqn_play_cartpole()
    # dqn_evaluate_policy()
