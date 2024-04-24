from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, MultiStep
from torchrl.envs import ParallelEnv, GymEnv, EnvCreator, TransformedEnv, Compose, StepCounter, ToTensorImage, \
    RewardScaling, GrayScale, Resize, CatFrames, ObservationNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type

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


def make_env(
        env_str,
        num_workers,
):
    parallel = num_workers > 1

    if parallel:
        base_env = ParallelEnv(
            4,
            EnvCreator(
                lambda: GymEnv(
                    env_str,
                    # from_pixels=True,
                    # pixels_only=True,
                    device=fetch_device(),
                )
            ),
        )
    else:
        base_env = GymEnv(
            env_str,
            # from_pixels=True,
            # pixels_only=True,
            device=fetch_device(),
        )

    return base_env


def atari_preprocessing(base_env, obs_norm_sd=None):
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
        stats,
        num_collectors,
        actor_explore,
        frames_per_batch,
        total_frames,
        device,
        gamma=0.99
):
    cls = SyncDataCollector

    if num_collectors > 1:
        cls = MultiaSyncDataCollector
    env_arg = [make_env(env_str=env_str, parallel=True, obs_norm_sd=stats)] * num_collectors
    data_collector = cls(
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
    return data_collector
