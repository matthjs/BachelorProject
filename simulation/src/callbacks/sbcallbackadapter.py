import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnNoModelImprovement

from callbacks.abstractcallback import AbstractCallback


class StableBaselinesCallbackAdapter(BaseCallback):
    """
    Adapter class such that my custom callbacks can be passed
    to the StableBaseline BaseAlgorithm .learn() method.
    NOTE: Might cause problems with parallel environment learning.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: AbstractCallback, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.callback = callback
        self.n_episodes = 0
        self._n_episodes_last = -1

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.callback.on_training_start()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"

        action_t = self.locals['actions']
        reward_t = self.locals['rewards']
        new_obs_t = self.locals['new_obs']
        done_t = self.locals['dones']

        if np.any(done_t):
            self.callback.on_episode_end()

        # iffy, results in moving data from GPU to CPU to GPU
        # maybe require that agents get tensors already.
        action = action_t.item() if len(action_t) == 1 else action_t
        reward = reward_t.item() if len(reward_t) == 1 else reward_t
        new_obs = new_obs_t[0] if len(new_obs_t) == 1 else new_obs_t
        done = done_t.item() if len(done_t) == 1 else done_t

        print(action, reward, new_obs, done)

        return self.callback.on_step(action, reward, new_obs, done)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.callback.on_training_end()
