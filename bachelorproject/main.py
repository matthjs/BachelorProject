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

def plot_cartpole():
    sim = SimulatorRL.load(experiment_id="experiment_CARTPOLE_THESIS_UPDATED")
    sim.plot_any_plottable_data()

def plot_lunar_lander():
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS")
    sim.plot_any_plottable_data()

"""
COLOR_LIST
-   DPQ (DGP) #FF7F50
-   GPQEGREEDY (DGP) #00FFFF
-   GPQUCB (DGP)    #000080
"""

if __name__ == "__main__":
    #plot_lunar_lander()
    # experiment_dummy_136, 143 maxes out reward signal for CartPole.
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_STRAT3")
    # 400 stopped at episode 110

    # sim = SimulatorRL("CartPole-v1", experiment_id="experiment_cartpole_archcompTRUEP")
    # b = Backupper(sim)  # backups experiment on SIGINT interrupt or normal exit.

    (sim
     #.register_agent("GPQ2 (SVGP)", "gpq_agent")
     # .register_agent("GPQ2 (DGP)", "gpq_agent")
     # .register_agent("GPQEGREEDY (DGP)", "gpq_agent")
     # .register_agent("GPQUCB (DGP)", "gpq_agent")
     # .register_agent("GPSARSALAPTOP (DGP)", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_2", "gpsarsa_agent")
     # .register_agent("sb_dqn_2", "sb_dqn")
     #.register_agent("DQN (MLP)", "sb_dqn")
     # .register_agent("sb_ppo_1", "sb_ppo")
     #.train_agents(agent_id_list=["random"], num_episodes=1000, concurrent=False,  callbacks=[RewardCallback(), UsageCallback()])
     # .train_agents(agent_id_list=["GPQEGREEDY (DGP)", "GPQUCB (DGP)"], num_episodes=1000, concurrent=False,
     #              callbacks=[EarlyStopCallback(RewardCallback(), 200, 5),
     #                         UsageCallback(),
     #                         LossCallback()])
     #.evaluate_agents(30, agent_id_list=["GPQEGREEDY (DGP)", "GPQUCB (DGP)"], callbacks=[RewardCallback(), UsageCallback()])
     #.data_to_csv()
     .plot_any_plottable_data(agent_id_list=["GPQ (DGP)", "GPQEGREEDY (DGP)", "GPQUCB (DGP)"], color_list=["#FF7F50", "#00FFFF", "#000080"]))
     #.save_agents()
     #)

    # sim.play("gpq_agent_3", 3)
    #sim.play("sb_dqn_1", 3)
    # sim.record("sb_dqn_1", 100)
