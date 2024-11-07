import pynvml
import icu_sepsis

from backup.backupper import Backupper
from callbacks.losscallback import LossCallback
from callbacks.rewardcallback import RewardCallback
from callbacks.usagecallback import UsageCallback
from simulators.simulator_rl import SimulatorRL

import gymnasium as gym
import icu_sepsis


print(gym.envs.registration.registry.keys())
env = gym.make('Sepsis/ICU-Sepsis-v2')


state, info = env.reset()
print('Initial state:', state)
print('Extra info:', info)

next_state, reward, terminated, truncated, info = env.step(0)
print('\nTaking action 0:')
print('Next state:', next_state)
print('Reward:', reward)
print('Terminated:', terminated)
print('Truncated:', truncated)