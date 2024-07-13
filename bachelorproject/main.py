import sys

import pynvml
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
    sim.plot_any_plottable_data(agent_id_list=["GPQ2 (DGP)", "DQN (MLP)", "sb_dqn_2", "gpq_agent_3", "random"],
                                color_list={"GPQ2 (DGP)": "#069af3", "DQN (MLP)": "#f97306", "sb_dqn_2": "#15b01a",
                                            "gpq_agent_3": "#7e1e9c", "random": "#ff81c0"})
    return sim


def play_cartpole():
    sim = SimulatorRL.load(experiment_id="experiment_CARTPOLE_THESIS_UPDATED")
    sim.record("sb_dqn_2", 1000)
    # sim.play("GPQ2 (DGP)", 5)
    # sim.play("DQN (MLP)", 5)
    # sim.evaluate_agents(30, agent_id_list=["GPQ2 (DGP)"], callbacks=[RewardCallback(), UsageCallback()])


def play_lunar_lander():
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_UPDATED+DEF")
    sim.record("DQN (Linear)", 10000)
    # sim.play("GPQ2 (DGP)", 5)
    # sim.play("DQN (MLP)", 5)
    # sim.evaluate_agents(30, agent_id_list=["GPQ2 (DGP)"], callbacks=[RewardCallback(), UsageCallback()])


def plot_lunar_lander():
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_UPDATED+DEF")
    sim.plot_any_plottable_data(agent_id_list=["GPQ2 (DGP)", "DQN (MLP)", "DQN (Linear)", "GPQ (SVGP)", "random"],
                                color_list={"GPQ2 (DGP)": "#069af3", "DQN (MLP)": "#f97306", "DQN (Linear)": "#15b01a",
                                            "GPQ (SVGP)": "#7e1e9c", "random": "#ff81c0"})
    return sim


def plot_lunar_lander2():
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_UPDATED+DEF")
    sim.plot_any_plottable_data(agent_id_list=["GPQ2 (DGP)", "GPQEGREEDY (DGP)", "GPQUCB (DGP)", "random"],
                                color_list={"GPQ2 (DGP)": "#069af3", "GPQEGREEDY (DGP)": "#ffff14",
                                            "GPQUCB (DGP)": "#e50000", "random": "#ff81c0"})
    return sim


def plot_lunar_lander3():
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_UPDATED+DEF")
    sim.plot_any_plottable_data(agent_id_list=["GPQ2 (DGP)", "GPSARSA (DGP)", "random"],
                                color_list={"GPQ2 (DGP)": "#069af3", "GPSARSA (DGP)": "#00ffff"})

    return sim

def plot_lunar_lander_def():
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_17")
    sim.plot_any_plottable_data(agent_id_list=["GPQ4 (DGP)", "GPSARSA (DGP)", "DQN2 (MLP)", "DQN2 (Linear)", "GPQ (SVGP)", "random"],
                                color_list={"GPQ4 (DGP)": "#069af3", "DQN2 (MLP)": "#f97306", "DQN2 (Linear)": "#15b01a",
                                            "GPQ (SVGP)": "#7e1e9c", "random": "#ff81c0", "GPSARSA (DGP)": "#00ffff"})



"""
COLOR_LIST
-   DPQ (DGP) #069af3
-   DPG2 (DGP) #069af3
-   DQN (MLP)   #f97306   RED-LIKE
-   DQN (Linear)    #15b01a
-   GPQEGREEDY (DGP) #ffff14
-   GPQUCB (DGP)    #e50000
-   GPQ (SVGP)  #7e1e9c
-   GPSARSA (DGP) #00ffff
-   GP-Q (GP)   #89fe05
"""

if __name__ == "__main__":
    plot_lunar_lander_def()
    sys.exit()
    pynvml.nvmlInit()
    # 1
    # plot_lunar_lander2()
    # plot_lunar_lander()
    # plot_lunar_lander()
    # plot_lunar_lander()
    # experiment_dummy_136, 143 maxes out reward signal for CartPole.
    # DO NOT USE +++
    # 3 is good.
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_16", new_experiment_id="experiment_LUNAR_THESIS_17")
    # 400 stopped at episode 110

    # sim = SimulatorRL("LunarLander-v2", experiment_id="experiment_LUNAR_THESIS2")
    b = Backupper(sim)  # backups experiment on SIGINT interrupt or normal exit.

    (sim
     # .register_agent("GPQ (SVGP)", "gpq_agent")
     .register_agent("GPQ4 (DGP)", "gpq_agent")
     # .register_agent("GPSARSA (DGP)", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_1", "gpsarsa_agent")
     # .register_agent("gpsarsa_agent_2", "gpsarsa_agent")
     # .register_agent("DQN2 (MLP)", "sb_dqn")
     # .register_agent("DQN2 (Linear)", "sb_dqn")
     # .register_agent("GPQEGREEDY (DGP)", "gpq_agent")
     # .register_agent("GPQUCB (DGP)", "gpq_agent")
     # .register_agent("sb_ppo_1", "sb_ppo")
     # .train_agents(agent_id_list=["random"], num_episodes=1000, concurrent=False,  callbacks=[RewardCallback(), UsageCallback()])
     .train_agents(agent_id_list=["GPQ4 (DGP)"],
                                  num_episodes=3000, concurrent=False,
                   callbacks=[EarlyStopCallback(RewardCallback(), 200, 5),
                              UsageCallback(),
                              LossCallback()])
     .evaluate_agents(30,
                      agent_id_list=[
                                  "GPQ4 (DGP)"], callbacks=[RewardCallback(), UsageCallback()])
     .data_to_csv()
     .plot_any_plottable_data()
     # .plot_any_plottable_data(agent_id_list=["GPSARSA2 (DGP)", "random"], color_list={"GPSARSA2 (DGP)":"#00ffff", "random":"#ff81c0"})
     # .plot_any_plottable_data(agent_id_list=["GPQ2 (DGP)", "DQN (MLP)", "DQN (Linear)", "GPQ (SVGP)", "random"],
     #                            color_list={"GPSARSA2 (DGP)": "#00ffff", "GPQ2 (DGP)": "#069af3",  "DQN (MLP)": "#f97306",  "DQN (Linear)": "#15b01a", "GPQ (SVGP)": "#7e1e9c", "random": "#ff81c0"})
     .save_agents()
     )

    """
    COLOR_LIST
    -   DPQ (DGP) #069af3
    -   DPG2 (DGP) #069af3
    -   DQN (MLP)   #f97306   RED-LIKE
    -   DQN (Linear)    #15b01a
    -   GPQEGREEDY (DGP) #ffff14
    -   GPQUCB (DGP)    #e50000
    -   GPQ (SVGP)  #7e1e9c
    -   GPSARSA (DGP) #00ffff
    -   GP-Q (GP)   #89fe05
    -   random      #ff81c0
    """

    # sim.play("GPQ2 (DGP)", 3)
    # sim.play("gpq_agent_3", 3)
    # sim.play("sb_dqn_1", 3)
    # sim.record("sb_dqn_1", 100)
    pynvml.nvmlShutdown()



