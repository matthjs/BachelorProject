from collections import namedtuple

import gymnasium as gym
import torch
from botorch.models.transforms import Normalize
from torchrl.data import ReplayBuffer, LazyTensorStorage


class Norm:
    """
    I am having some issues with mixedGP transform function I do not know why.
    """

    def __init__(self, in_dim: int, cat_ind: int = 1):
        self.normalizer = Normalize(d=in_dim)
        self.reverse_normalizer = Normalize(d=in_dim, reverse=True)
        self.cat_ind = cat_ind

    def transform(self, data_x: torch.tensor):
        continuous_features = data_x[:, :-self.cat_ind]
        continuous_transformed = self.normalizer(continuous_features)
        categorical_features = data_x[:, -self.cat_ind:]
        return torch.cat([continuous_transformed, categorical_features], dim=1)

    def untransform(self, data_x):
        continuous_features = data_x[:, :-self.cat_ind]
        continuous_transformed = self.reverse_normalizer(continuous_features)
        categorical_features = data_x[:, -self.cat_ind:]
        return torch.cat([continuous_transformed, categorical_features], dim=1)


def test_e():
    env = gym.make("MountainCar-v0", render_mode="human")
    obs, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        print(action)
        print(type(action))
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def test_buf_again():
    env = gym.make("MountainCar-v0")

    obs, info = env.reset()

    buf = ReplayBuffer(storage=LazyTensorStorage(20, device="cuda"))

    for cnt in range(10):
        action = env.action_space.sample()
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        print("tau: ", (old_obs, cnt, reward, obs))
        state_t = torch.as_tensor(old_obs, device="cuda")
        action_t = torch.as_tensor(cnt, device="cuda")
        reward_t = torch.as_tensor(reward, device="cuda")
        next_state_t = torch.as_tensor(obs, device="cuda")

        print("DEV ->", state_t.device)

        buf.add((state_t, action_t, reward_t, next_state_t))

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    trajectories = buf.sample(batch_size=2)

    print("state: ", trajectories[0].device)
    print("action: ", trajectories[1])
    print("reward: ", trajectories[2])
    print("next_state: ", trajectories[3])
