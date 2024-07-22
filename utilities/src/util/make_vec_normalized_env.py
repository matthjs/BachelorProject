import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Dict, Union

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class CustomVecNormalize(VecNormalize):
    """
    Custom vectorized environment normalization for specific observation indices.
    """

    def __init__(self,
                 venv,
                 training=True,
                 norm_obs=True,
                 norm_reward=True,
                 clip_obs=10.0,
                 clip_reward=10.0,
                 gamma=0.99,
                 epsilon=1e-8,
                 norm_obs_indices=0):
        """
        :param venv: The vectorized environment.
        :param training: Whether the environment is in training mode. Default is True.
        :param norm_obs: Whether to normalize observations. Default is True.
        :param norm_reward: Whether to normalize rewards. Default is True.
        :param clip_obs: The clipping value for observations. Default is 10.0.
        :param clip_reward: The clipping value for rewards. Default is 10.0.
        :param gamma: The discount factor for reward normalization. Default is 0.99.
        :param epsilon: A small value to avoid division by zero during normalization. Default is 1e-8.
        :param norm_obs_indices: Number of observation indices to normalize. Default is 0 (normalize all).
        """
        super().__init__(venv,
                         training=training,
                         norm_obs=norm_obs,
                         norm_reward=norm_reward,
                         clip_obs=clip_obs,
                         clip_reward=clip_reward,
                         gamma=gamma,
                         epsilon=epsilon)

        self.norm_obs_indices = norm_obs_indices
        if norm_obs_indices > 0:
            # Initialize RunningMeanStd only for the first N indices
            self.obs_rms = RunningMeanStd(shape=(norm_obs_indices,))
        else:
            # Initialize a single RunningMeanStd for the whole observation space
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.

        :param obs: Observations to be normalized.
        :param obs_rms: Running mean and standard deviation statistics.
        :return: Normalized observations.
        """
        if self.norm_obs_indices > 0:
            # Normalize only the first N indices
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)  # I do not know why I need to do this.
            normalized_obs = obs.copy()
            normalized_obs[:, :self.norm_obs_indices] = np.clip(
                (obs[:, :self.norm_obs_indices] - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs,
                self.clip_obs)
            return normalized_obs
        else:
            # Normalize the whole observation space
            return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to unnormalize observation.

        :param obs: Normalized observations to be unnormalized.
        :param obs_rms: Running mean and standard deviation statistics.
        :return: Unnormalized observations.
        """
        if self.norm_obs_indices > 0:
            # Unnormalize only the first N indices
            # unnormalized_obs = obs
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)  # I do not know why I need to do this.
            unnormalized_obs = obs.copy()
            unnormalized_obs[:, :self.norm_obs_indices] = (obs[:, :self.norm_obs_indices] * np.sqrt(
                obs_rms.var + self.epsilon)) + obs_rms.mean
            return unnormalized_obs
        else:
            # Unnormalize the whole observation space
            return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

    def update_rms_obs(self, obs):
        """
        Update the running mean and standard deviation for the specified observation indices.

        :param obs: Observations to update statistics with.
        """
        if self.norm_obs_indices > 0:
            self.obs_rms.update(obs[:, :self.norm_obs_indices])
        else:
            self.obs_rms.update(obs)

    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, (np.ndarray, dict))  # for mypy
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.update_rms_obs(obs)  # CHANGED

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        :return: First observation of the episode
        """
        obs = self.venv.reset()
        assert isinstance(obs, (np.ndarray, dict))
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                assert isinstance(self.obs_rms, RunningMeanStd)
                self.update_rms_obs(obs)  # CHANGED
        return self.normalize_obs(obs)


def make_vec_env(env_id: str):
    return DummyVecEnv([lambda: gym.make(env_id, render_mode='rgb_array')])


def make_vec_normalized_env(env_str: str,
                            training: bool = True,
                            norm_obs: bool = True,
                            norm_reward: bool = False,
                            clip_obs: float = 10.0,
                            clip_reward: float = 10.0,
                            gamma: float = 0.99,
                            epsilon: float = 1e-08) -> VecNormalize:
    """
    Create a VecNormalize environment for the specified Gym environment string.

    :param env_str: The string identifier for the Gym environment.
    :param training: Whether the environment is for training. Default is True.
    :param norm_obs: Whether to normalize observations. Default is True.
    :param norm_reward: Whether to normalize rewards. Default is False.
    :param clip_obs: The clipping value for observations. Default is 10.0.
    :param clip_reward: The clipping value for rewards. Default is 10.0.
    :param gamma: The discount factor for reward normalization. Default is 0.99.
    :param epsilon: A small value to avoid division by zero during normalization. Default is 1e-08.
    :return: A VecNormalize environment with the specified settings.
    """
    norm_indices = 0
    if env_str == "LunarLander-v2":
        norm_indices = 6  # 8 - 2 = 6 We should not normalize the last two dimensions.

    vec_env = make_vec_env(env_str)
    vec_normalized_env = CustomVecNormalize(venv=vec_env,
                                            training=training,
                                            norm_obs=norm_obs,
                                            norm_reward=norm_reward,
                                            clip_obs=clip_obs,
                                            clip_reward=clip_reward,
                                            gamma=gamma,
                                            epsilon=epsilon,
                                            norm_obs_indices=norm_indices)
    return vec_normalized_env
