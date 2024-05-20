import gymnasium as gym


class InitialStateWrapper(gym.Wrapper):
    """
    A wrapper for Gym environments that allows setting the initial state.

    This wrapper modifies the reset() method to accept an optional parameter for the initial state.
    It sets the initial state of the environment if provided, otherwise behaves as the original reset() method.

    :param env: The Gym environment to wrap.
    :type env: gym.Env

    :ivar initial_state: The initial state of the environment.
    :vartype initial_state: object
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the InitialStateWrapper.

        :param env: The Gym environment to wrap.
        :type env: gym.Env
        """
        super().__init__(env)
        self.initial_state = None

    def reset(self, initial_state=None, **kwargs):
        """
        Reset the environment.

        :param initial_state: The initial state to set the environment to.
        :type initial_state: object, optional
        :param kwargs: Additional arguments to pass to the underlying reset() method.
        :return: The initial observation from the environment.
        :rtype: object
        """
        if initial_state is None:
            return self.env.reset(**kwargs)
        else:
            self.initial_state = initial_state
            self.env.reset(**kwargs)
            self.env.unwrapped.state = initial_state  # Assuming environment has an attribute 'state'
            return initial_state

    def step(self, action):
        """
        Take a step in the environment.

        :param action: The action to take in the environment.
        :type action: object
        :return: A tuple containing the next observation, the reward, a flag indicating termination, and additional information.
        :rtype: tuple
        """
        return self.env.step(action)
