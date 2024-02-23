import gymnasium as gym
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs import GymEnv


def env_interaction_gym(agent_type: str, env: gym.Env, setting: str, episodes: int):
    obs, info = env.reset()

    for _ in range(episodes):
        action = env.action_space.sample()
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    print(buf)
    env.close()


def env_interaction(agent_type: str, env_str: str, setting: str, episodes: int, render_mode: str):
    env = gym.make(env_str, render_mode=render_mode)

    env_interaction_gym(agent_type, env, setting, episodes)


def gym_env_interact(agent_type: str, env_str: str, episodes: int):
    env = GymEnv(env_str).to("cuda")
    rollout = env.rollout(max_steps=10)
    print(rollout)
