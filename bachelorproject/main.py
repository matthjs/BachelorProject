import torch

from gp.bayesianoptimizer_rl import append_actions
from loops.envinteraction import env_interaction_gym, env_interaction_gym2
from loops.stablebaselinesrun import dqn_train, dqn_evaluate_policy, dqn_train2, dqn_play_cartpole
from simulators.simulator_rl import SimulatorRL

if __name__ == "__main__":
    sim = SimulatorRL("CartPole-v1")
    (sim.register_agent("gpq_agent_1", "gpq_agent")
     .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     .register_agent("sb_dqn", "sb_dqn")
     .train_agents(num_episodes=10, concurrent=False, logging=True)
     .evaluate_agents("CartPole-v1", 10)
     .data_to_csv()
     .plot_any_plottable_data())

    sim.play("gpq_agent_1", 10)
