from loguru import logger

from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL


def test_load_experiment():
    logger.debug("loading experiment...")

    (SimulatorRL("CartPole-v1")
     .load_agent("gpq_agent_1", "gpq_agent")
     .load_agent("gpsarsa_agent_1", "gpsarsa_agent")
     .load_agent("sb_dqn_1", "sb_dqn")
     .load_agent("sb_ppo_1", "sb_ppo")
     .evaluate_agents("CartPole-v1", 10,
                      callbacks=[RewardCallback(), UsageCallback()]))


if __name__ == "__main__":
    sim = SimulatorRL("CartPole-v1", experiment_id="experiment!")
    (sim
     .register_agent("gpq_agent_1", "gpq_agent")
     .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     #  .register_agent("sb_dqn_1", "sb_dqn")
     .register_agent("sb_ppo_1", "sb_ppo")
     .train_agents(num_episodes=50, concurrent=False,
                   callbacks=[RewardCallback(), UsageCallback()])
     .evaluate_agents("CartPole-v1", 10,
                      callbacks=[RewardCallback(), UsageCallback()])
     .data_to_csv()
     .plot_any_plottable_data()
     .save_agents())

    sim.play("gpq_agent_1", 10)

    # test_load_experiment()
