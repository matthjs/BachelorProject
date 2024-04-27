import torch
from loguru import logger
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, MultiStep
from torchrl.envs import ParallelEnv, GymEnv, EnvCreator, TransformedEnv, Compose, StepCounter, ToTensorImage, \
    RewardScaling, GrayScale, Resize, CatFrames, ObservationNorm, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record import VideoRecorder

from util.fetchdevice import fetch_device


def get_replay_buffer(buffer_size: int, n_optim: int, batch_size: int) -> TensorDictReplayBuffer:
    """
    :param buffer_size: The maximum size of the replay buffer.
    :param n_optim: Number of parallel processes performing optimization.
    :param batch_size: Batch size used for sampling from the replay buffer during training.
    :return: The constructed replay buffer.
    """
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size),
        prefetch=n_optim,
    )
    return replay_buffer

def make_env(env_name="MountainCar-v0", device="cpu", from_pixels=False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env

def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()

def make_env2(
        env_str,
        num_workers=1,
):
    parallel = num_workers > 1

    if parallel:
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(
                lambda: GymEnv(
                    env_str,
                    from_pixels=True,
                    pixels_only=True,
                    device=fetch_device(),
                )
            ),
        )
    else:
        base_env = GymEnv(
            env_str,
            from_pixels=True,
            pixels_only=True,
            device=fetch_device(),
        )

    return base_env

def atari_preprocessing(env_str, num_workers):
    obs_norm_sd = None # get_norm_stats(env_str)

    base_env = make_env2(env_str, num_workers)

    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}

    return TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        )
    )


def get_collector(
        env_str,
        env_fun,
        num_workers,
        num_collectors,
        actor_explore,
        frames_per_batch,
        total_frames,
        device,
        gamma,
):
    logger.debug("getting collector")
    env_arg = [env_fun(env_str, num_workers)] * num_collectors
    data_collector = MultiaSyncDataCollector(
        env_arg,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )

    logger.debug("finished constructing datacollector.")
    return data_collector


def get_norm_stats(env_str):
    test_env = make_env(env_str)
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    test_env.close()
    return obs_norm_sd
