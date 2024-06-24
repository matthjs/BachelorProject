from loguru import logger
from backup.backupper import Backupper
from callbacks.earlystopcallback import EarlyStopCallback
from callbacks.losscallback import LossCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL


def test_load_experiment():
    logger.debug("loading experiment...")

    (SimulatorRL("CartPole-v1", experiment_id="experiment_dummy_116")  # 75
     # .load_agent("gpq_agent_1", "gpq_agent")
     .load_agent("gpq_agent_1", "gpq_agent")
     # .load_agent("sb_dqn_1", "sb_dqn")
     # .load_agent("sb_ppo_1", "sb_ppo")
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .play("gpq_agent_1", 20))


if __name__ == "__main__":
    # experiment_dummy_136, 143 maxes out reward signal for CartPole.
    # sim = SimulatorRL.load(experiment_id="experiment_dummy_272", new_experiment_id="experiment_dummy_273")

    sim = SimulatorRL("CartPole-v1", experiment_id="experiment_dummy_305")
    b = Backupper(sim)  # backups experiment on SIGINT interrupt or normal exit.

    (sim
     .register_agent("gpq_agent_3", "gpq_agent")
     # .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_2", "gpsarsa_agent")
     #.register_agent("sb_dqn_1", "sb_dqn")
     # .register_agent("sb_dqn_2", "sb_dqn")
     # .register_agent("sb_ppo_1", "sb_ppo")
     .train_agents(agent_id_list=["random"], num_episodes=100, concurrent=False,  callbacks=[RewardCallback(), UsageCallback()])
     .train_agents(agent_id_list=["gpq_agent_3"], num_episodes=1000, concurrent=False,
                   callbacks=[EarlyStopCallback(RewardCallback(), 500, 5),
                              UsageCallback(),
                              LossCallback()])
     #.train_agents(num_episodes=1000, concurrent=False,
     #              callbacks=[RewardCallback(), UsageCallback(), LossCallback()])
     .evaluate_agents(30, callbacks=[RewardCallback(), UsageCallback()])
     .data_to_csv()
     .plot_any_plottable_data()
     .save_agents()
     )

    sim.play("gpq_agent_3", 3)
    # sim.record("sb_dqn_1", 100)
