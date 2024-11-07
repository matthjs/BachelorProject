import pynvml
import icu_sepsis
from bachelorproject.simulation.backup.backupper import Backupper
from bachelorproject.simulation.callbacks.losscallback import LossCallback
from bachelorproject.simulation.callbacks.rewardcallback import RewardCallback
from bachelorproject.simulation.callbacks.usagecallback import UsageCallback
from bachelorproject.simulation.simulators.simulator_rl import SimulatorRL
from gymnasium.envs.registration import register

# Code does not work well with env_ids that include paths '/'
register(
    id='ICU-Sepsis-v2',
    entry_point='icu_sepsis:icu_sepsis_flat',
    max_episode_steps=500)

if __name__ == "__main__":
    pynvml.nvmlInit()
    sim = SimulatorRL("ICU-Sepsis-v2", experiment_id="experiment_sepsis")
    back_upper = Backupper(sim)  # backups experiment on SIGINT interrupt or normal exit.

    # You may want to train each agent sequentially in separate processes to get more accurate
    # VRAM usage estimates.
    (sim
     .register_agent("GPQ (DGP)", "gpq_agent")
     .train_agents(num_episodes=300, concurrent=False,
                   callbacks=[UsageCallback(),
                              LossCallback()])
     .evaluate_agents(30, callbacks=[RewardCallback(), UsageCallback()])
     .plot_any_plottable_data()
     .data_to_csv()
     )
    pynvml.nvmlShutdown()
