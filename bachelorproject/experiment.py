from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL

if __name__ == "__main__":
    sim = SimulatorRL("CartPole-v1", experiment_id="experiment_dummy_82")
    (sim
     .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     .register_agent("sb_dqn_1", "sb_dqn")
     .register_agent("sb_dqn_2", "sb_dqn")
     .train_agents(num_episodes=100, concurrent=False,
                   callbacks=[RewardCallback(), UsageCallback()])
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .data_to_csv()
     .plot_any_plottable_data()
     .save_agents())

    sim.play("gpsarsa_agent_1", 10)
