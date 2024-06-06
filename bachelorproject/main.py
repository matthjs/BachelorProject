from loguru import logger

from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL


def test_load_experiment():
    logger.debug("loading experiment...")

    (SimulatorRL("LunarLander-v2", experiment_id="experiment_dummy_94")        # 75
     # .load_agent("gpq_agent_1", "gpq_agent")
     .load_agent("gpq_agent_2", "gpq_agent")
     # .load_agent("sb_dqn_1", "sb_dqn")
     # .load_agent("sb_ppo_1", "sb_ppo")
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .play("gpq_agent_2", 20))


if __name__ == "__main__":
    sim = SimulatorRL("CartPole-v1", experiment_id="experiment_dummy_105")
    (sim
     # .register_agent("linear_q_agent_1", "linear_q_agent")
     .register_agent("gpq_agent_2", "gpq_agent")
     # .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_2", "gpsarsa_agent")
     .register_agent("sb_dqn_1", "sb_dqn")
     .register_agent("sb_dqn_2", "sb_dqn")
     # .register_agent("sb_ppo_1", "sb_ppo")
     .train_agents(num_episodes=300, concurrent=False,
                   callbacks=[RewardCallback(), UsageCallback()])
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .data_to_csv()
     .plot_any_plottable_data()
     .save_agents())

    sim.play("gpq_agent_2", 20)
