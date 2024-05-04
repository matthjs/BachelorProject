import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from metricstracker.metricstracker import MetricsTracker


def dqn_train(env_str='MountainCar-v0', total_time_step=int(1.2e5)):
    env = gym.make(env_str)

    model = DQN(
        "MlpPolicy",  # What model to use to approximate Q-function.
        env,
        gamma=0.98,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,  # epsilon-greedy schedule
        learning_rate=4e-3,
        batch_size=128,
        learning_starts=100000,
        target_update_interval=600,
        buffer_size=10000,  # Replay buffer size
        optimize_memory_usage=False,
        policy_kwargs=dict(net_arch=[256, 256])
    )
    model.learn(total_timesteps=total_time_step)

    model.save("dqn_mountain_car")

    env.close()

def dqn_train2(env_str="CartPole-v1", total_time_step=int(5e4)):
    env = gym.make(env_str)

    model = DQN(
        "MlpPolicy",  # What model to use to approximate Q-function.
        env,
        gamma=0.99,
        verbose=1,
        train_freq=256,
        gradient_steps=128,
        exploration_fraction=0.16,
        exploration_final_eps=0.04,  # epsilon-greedy schedule
        learning_rate=2.3e-3,
        batch_size=64,
        learning_starts=1000,
        target_update_interval=10,
        buffer_size=100000,  # Replay buffer size
        policy_kwargs=dict(net_arch=[256, 256])
    )
    model.learn(total_timesteps=total_time_step)

    model.save("dqn_cartpole-v1")

    env.close()


# Set up fake display; otherwise rendering will fail
import os

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


# !apt-get install ffmpeg freeglut3-dev xvfb  -y # For visualization

def record_video(eval_env, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param eval_env: environment.
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env.metadata['render_fps'] = 60
    vec_env_record = VecVideoRecorder(eval_env,
                                      video_folder,
                                      record_video_trigger=lambda x: x == 0,
                                      # Function that defines when to start recording.
                                      video_length=video_length,
                                      name_prefix=prefix)

    obs = vec_env_record.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = vec_env_record.step(action)

    # Close the video recorder
    vec_env_record.close()


import base64
from pathlib import Path

from IPython import display as ipythondisplay


def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def dqn_play_cartpole(time_steps=3000):
    agent = DQN.load("./dqn_cartpole-v1.zip")

    test_env = make_vec_env("CartPole-v1", n_envs=1)

    record_video(test_env, agent, video_length=5000, prefix="dqn-mc")
    show_videos("videos", prefix="dqn-mc")


def dqn_evaluate_policy(agent_str="./dqn_cartpole-v1.zip", env_str="CartPole-v1", n_eval_ep=10):
    model = DQN.load(agent_str)

    test_env = make_vec_env(env_str, n_envs=1, seed=0)
    # Frame-stacking with 4 frames
    # test_env = VecFrameStack(test_env, n_stack=4)

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=n_eval_ep, warn=False)

    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
