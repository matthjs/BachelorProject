from threading import current_thread

import gymnasium as gym
from loguru import logger
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs import GymEnv

from agentfactory.agentfactory import AgentFactory
from metricstracker.metricstracker import MetricsTracker


def env_interaction_gym(agent_type: str, env_str: str, time_steps: int, render_mode: str = "human"):
    env = gym.make(env_str)
    obs, info = env.reset()
    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type, env_str=env_str)

    episode_reward = 0

    for _ in range(time_steps):
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.record_env_info(info, terminated or truncated)
        agent.add_trajectory((old_obs, action, reward, obs))

        episode_reward += reward

        agent.update()

        if terminated or truncated:
            logger.debug(f"Episode reward: {episode_reward}")
            MetricsTracker().record_return(current_thread().name, episode_reward)
            episode_reward = 0
            obs, info = env.reset()

    env.close()
    print(MetricsTracker().loss_history)
    print(MetricsTracker().return_history)
    MetricsTracker().plot()


def env_interaction_gym2(agent_type: str, env_str: str, num_episodes: int, render_mode: str = "human"):
    env = gym.make(env_str)
    obs, info = env.reset()
    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type, env_str=env_str)

    episode_reward = 0
    highest_average_return: float = -999

    while True:
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.record_env_info(info, terminated or truncated)
        agent.add_trajectory((old_obs, action, reward, obs))

        episode_reward += reward

        agent.update()

        if terminated or truncated:
            num_episodes -= 1
            logger.debug(f"Episode reward: {episode_reward}")
            MetricsTracker().record_return(agent_type, episode_reward)
            episode_reward = 0
            obs, info = env.reset()

            current_avg_return: float = MetricsTracker().latest_average_return(agent_type)
            if current_avg_return > highest_average_return:
                highest_average_return = current_avg_return
                # agent.save()
                MetricsTracker().plot_return(title=env_str + "_" + agent_type)

        if num_episodes == 0:
            break

    env.close()
    print(MetricsTracker().loss_history)
    print(MetricsTracker().return_history)
    MetricsTracker().plot_return(title=env_str + "_" + agent_type)
