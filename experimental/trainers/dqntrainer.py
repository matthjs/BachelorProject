from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from experimental.common.rlutil import make_env
from experimental.trainers.abstracttrainer import AbstractTrainer


class DQNTrainer(AbstractTrainer):
    def __init__(self,
                 env_name,
                 exploration_policy,
                 frames_per_batch,
                 total_frames,
                 init_random_frames,
                 rb_buffer_size,
                 rb_batch_size,
                 device="cpu",
                 storing_device="cpu",
                 ):
        collector = SyncDataCollector(
            create_env_fn=make_env(env_name, device),
            policy=exploration_policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=storing_device,
            max_frames_per_traj=-1,
            init_random_frames=init_random_frames,
        )

        # Create the replay buffer
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=10,
            storage=LazyTensorStorage(
                max_size=rb_batch_size,
                device=device,
            ),
            batch_size=rb_batch_size,
        )

