import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from callbacks.abstractcallback import AbstractCallback


class StableBaselinesCallbackAdapter(BaseCallback):
    """
    Adapter class to use custom callbacks with Stable Baselines models.

    This class allows custom callbacks to be passed to the StableBaseline BaseAlgorithm .learn() method.
    NOTE: Might cause problems with parallel environment learning.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
    """

    def __init__(self, callback: AbstractCallback, verbose: int = 0):
        """
        Constructor for StableBaselinesCallbackAdapter.

        :param callback: The custom callback to adapt.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.callback = callback
        self.n_episodes = 0
        self._n_episodes_last = -1

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.callback.on_training_start()

    def on_rollout_start(self) -> None:
        """
        This method is called before a rollout starts.
        """
        self.callback.on_update_start()

    def on_rollout_end(self) -> None:
        """
        This method is called after a rollout ends.
        """
        self.callback.on_update_end()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"

        action_t = self.locals['actions']
        reward_t = self.locals['rewards']
        new_obs_t = self.locals['new_obs']
        done_t = self.locals['dones']

        if np.any(done_t):
            self.callback.on_episode_end()

        action = action_t.item() if len(action_t) == 1 else action_t
        reward = reward_t.item() if len(reward_t) == 1 else reward_t
        new_obs = new_obs_t[0] if len(new_obs_t) == 1 else new_obs_t
        done = done_t.item() if len(done_t) == 1 else done_t

        return self.callback.on_step(action, reward, new_obs, done)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.callback.on_training_end()