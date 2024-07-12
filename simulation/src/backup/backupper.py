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
        self.triggered = False
        atexit.register(self.backup_experiment)
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        if not self.triggered:
            logger.info("Exiting program... ")
            # Perform any cleanup actions here
            self.simulator.evaluate_agents(30,
                                           agent_id_list=[
                                               "GPQ3 (DGP)"], callbacks=[RewardCallback(), UsageCallback()])
            self.simulator.data_to_csv()
            self.simulator.plot_any_plottable_data()
            self.backup_experiment()  # Call your method or perform any necessary actions
            self.triggered = True
            sys.exit(-1)

    def backup_experiment(self) -> None:
        if not self.triggered:
            logger.info("backing up experiment...")
            self.simulator.data_to_csv()
            self.simulator.save()
            self.triggered = True
