from collections import namedtuple

import gymnasium as gym
import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules import EGreedyModule

from loops.envinteraction import env_interaction_gym


def test_memory_buffer():
    buf = ReplayBuffer(storage=LazyTensorStorage(10))
    buf.add(torch.tensor((1, 2, 3, 4)).to("cuda"))
    buf.add(torch.tensor((4, 5, 6, 7)).to("cuda"))
    trajectories = buf.sample(batch_size=2)

    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state'))

    batch = Transition(*zip(*trajectories))
    print(batch.next_state)


def test():
    pass

def testE():
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

def testBufAgain():
    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state'))
    env = gym.make("MountainCar-v0")

    obs, info = env.reset()

    buf = ReplayBuffer(storage=LazyTensorStorage(20))

    for cnt in range(10):
        action = env.action_space.sample()
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        print("tau: ", (old_obs, cnt, reward, obs))
        state_t = torch.as_tensor(old_obs, device="cuda")
        action_t = torch.as_tensor(cnt, device="cuda")
        reward_t = torch.as_tensor(reward, device="cuda")
        next_state_t = torch.as_tensor(obs, device="cuda")

        buf.add((state_t, action_t, reward_t, next_state_t))

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    trajectories = buf.sample(batch_size=2)

    print("state: ", trajectories[0])
    print("action: ", trajectories[1])
    print("reward: ", trajectories[2])
    print("next_state: ", trajectories[3])



if __name__ == "__main__":
    testBufAgain()
    #env_interaction_gym("mlp_q_agent", "MountainCar-v0", 1000)
    """
    MetricsTracker().record_loss("cat", 2)
    MetricsTracker().record_loss("cat", 4)
    MetricsTracker().record_reward("cat", 24)
    MetricsTracker().record_reward("cat", 2)
    print(MetricsTracker().loss_history["cat"])
    print(MetricsTracker().reward_history["cat"])
    """
