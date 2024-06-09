import signal

from loguru import logger

from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL
import atexit
import sys


class Backupper:
    def __init__(self, simulator: SimulatorRL = None):
        self.simulator = simulator
        atexit.register(self.backup_experiment)
        signal.signal(signal.SIGINT, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        logger.info("Exiting program... ")
        # Perform any cleanup actions here
        self.simulator.evaluate_agents(10, callbacks=[RewardCallback(), UsageCallback()])
        self.simulator.data_to_csv()
        self.simulator.plot_any_plottable_data()
        self.backup_experiment()  # Call your method or perform any necessary actions
        sys.exit(-1)

    def backup_experiment(self) -> None:
        logger.info("backing up experiment...")
        self.simulator.save()
