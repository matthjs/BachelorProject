from threading import current_thread

import gymnasium as gym
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs import GymEnv

from agentfactory.agentfactory import AgentFactory
from metricstracker.metricstracker import MetricsTracker


def env_interaction_gym(agent_type: str, env_str: str, time_steps: int, render_mode: str = "human"):
    env = gym.make(env_str, render_mode=render_mode)
    obs, info = env.reset()
    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type, env=env)

    for _ in range(time_steps):
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        MetricsTracker().record_reward(current_thread().name, reward)
        agent.record_env_info(info, terminated or truncated)
        agent.add_trajectory((old_obs, action, reward, obs))

        agent.update()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(MetricsTracker().loss_history)
    print(MetricsTracker().reward_history)
    MetricsTracker().plot()


def gym_env_interact(agent_type: str, env_str: str, episodes: int):
    env = GymEnv(env_str).to("cuda")
    rollout = env.rollout(max_steps=10)
    print(rollout)
