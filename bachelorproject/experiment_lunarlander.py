import pynvml

from backup.backupper import Backupper
from callbacks.earlystopcallback import EarlyStopCallback
from callbacks.losscallback import LossCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL

if __name__ == "__main__":
    pynvml.nvmlInit()
    sim = SimulatorRL("CartPole-v1", experiment_id="experiment_cartpole")
    back_upper = Backupper(sim)  # backups experiment on SIGINT interrupt or normal exit.

    # You may want to train each agent sequentially in separate processes to get more accurate
    # VRAM usage estimates.
    (sim
     .register_agent("GPQ (SVGP)", "gpq_agent")
     .register_agent("GPQ (DGP)", "gpq_agent")
     .register_agent("DQN (MLP)", "sb_dqn")
     .register_agent("DQN (Linear)", "sb_dqn")
     .train_agents(num_episodes=10, concurrent=False,
                   callbacks=[EarlyStopCallback(RewardCallback(), 500, 5),
                              UsageCallback(),
                              LossCallback()])
     .evaluate_agents(30, callbacks=[RewardCallback(), UsageCallback()])
     .plot_any_plottable_data()
     .data_to_csv()
     )
    pynvml.nvmlShutdown()