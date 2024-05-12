from stable_baselines3 import DQN

from agent.abstractagent import AbstractAgent
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork


class StableBaselinesDQNAgent(AbstractAgent):
    def __init__(self, policy: Union[str, Type[DQNPolicy]], env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 1e-4, buffer_size: int = 1_000_000,
                 learning_starts: int = 50000, batch_size: int = 32, tau: float = 1.0, gamma: float = 0.99,
                 train_freq: Union[int, Tuple[int, str]] = 4, gradient_steps: int = 1,
                 replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None, optimize_memory_usage: bool = False,
                 target_update_interval: int = 10000, exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.05, max_grad_norm: float = 10,
                 stats_window_size: int = 100, tensorboard_log: Optional[str] = None,
                 policy_kwargs: Optional[Dict[str, Any]] = None, verbose: int = 0, seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto", _init_setup_model: bool = True) -> None:
        super().__init__({}, env.observation_space, env.action_space)
        self.model = DQN(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )

    def add_trajectory(self, trajectory: tuple) -> None:
        print("This is a StableBaselinesWrapper so this does nothing!")

    def update(self):
        pass

    def policy(self, state):
        pass

    def is_stable_baselines_wrapper(self):
        return True
