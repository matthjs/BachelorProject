import pynvml
from callbacks.earlystopcallback import EarlyStopCallback
from callbacks.losscallback import LossCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL


def plot_cartpole():
    sim = SimulatorRL.load(experiment_id="experiment_CARTPOLE_THESIS_UPDATED")
    sim.plot_any_plottable_data(agent_id_list=["GPQ2 (DGP)", "DQN (MLP)", "sb_dqn_2", "gpq_agent_3", "random"],
                                color_list={"GPQ2 (DGP)": "#069af3", "DQN (MLP)": "#f97306", "sb_dqn_2": "#15b01a",
                                            "gpq_agent_3": "#7e1e9c", "random": "#ff81c0"})
    return sim


def play_cartpole(agent_id, num_timesteps=5000):
    sim = SimulatorRL.load(experiment_id="experiment_CARTPOLE_THESIS_UPDATED")
    sim.record(agent_id, num_timesteps)


def play_lunar_lander(agent_id, num_timesteps=10000):
    sim = SimulatorRL.load(experiment_id="experiment_LUNAR_THESIS_17")
    sim.record(agent_id, num_timesteps)


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
    sim.plot_any_plottable_data(
        agent_id_list=["GPQ4 (DGP)", "GPSARSA (DGP)", "DQN2 (MLP)", "DQN2 (Linear)", "GPQ (SVGP)", "random"],
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
    pynvml.nvmlInit()
    sim = SimulatorRL("CartPole-v1", experiment_id="X")

    (sim
     .register_agent("GPQ (DGP)", "gpq_agent")
     .train_agents(num_episodes=10, concurrent=False,
                   callbacks=[EarlyStopCallback(RewardCallback(), 500, 5),
                              UsageCallback(),
                              LossCallback()])
     .evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
     .plot_any_plottable_data()
     )

    pynvml.nvmlShutdown()
