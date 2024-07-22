import signal

from loguru import logger

from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL
import atexit
import sys


class Backupper:
    """
    Class to handle backup operations for a SimulatorRL instance upon receiving termination signals.
    """

    def __init__(self, simulator: SimulatorRL = None) -> None:
        """
        Constructor for Backupper.

        :param simulator: Instance of SimulatorRL to be backed up.
        """
        self.simulator = simulator
        self.triggered = False
        atexit.register(self.backup_experiment)
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame) -> None:
        """
        Handle termination signals to perform a graceful exit.

        :param signum: Signal number.
        :param frame: Current stack frame.
        """
        if not self.triggered:
            logger.info("Exiting program... ")
            self.simulator.evaluate_agents(30, callbacks=[RewardCallback(), UsageCallback()])
            self.simulator.data_to_csv()
            self.simulator.plot_any_plottable_data()
            self.backup_experiment()
            self.triggered = True
            sys.exit(-1)

    def backup_experiment(self) -> None:
        """
        Backup the experiment data and save the simulator state.
        """
        if not self.triggered:
            logger.info("Backing up experiment...")
            self.simulator.data_to_csv()
            self.simulator.save()
            self.triggered = True
