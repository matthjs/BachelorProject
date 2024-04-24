import tempfile
import uuid
import warnings

from loguru import logger
from tensordict.nn import TensorDictSequential, TensorDictModule
from torch import nn
from torch.optim import Adam
from torchrl.modules import QValueActor, QValueModule, EGreedyModule, DuelingCnnDQNet
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record import CSVLogger
from torchrl.trainers import Trainer

from bachelorproject.common.fetchdevice import fetch_device
from bachelorproject.common.rlutil import make_env, get_collector, get_norm_stats, atari_preprocessing, \
    get_replay_buffer


class DQNAgent:
    def __init__(self,
                 env_str,
                 lr=2e-3,
                 wd=1e-5,
                 betas=(0.9, 0.999),
                 n_optim=8,
                 gamma=0.99,
                 tau=0.02,
                 total_frames=5000,
                 init_random_frames=100,
                 frames_per_batch=32,
                 batch_size=32,
                 max_buffer_size=100000,
                 num_workers=2,
                 num_collectors=2,
                 eps_greedy_val=0.1,
                 eps_greedy_val_env=0.005,
                 env_fun=atari_preprocessing):
        """
        Class for creating components of DQN agent.

        :param env_str: The environment name or string identifier.
        :param lr: Learning rate for the optimizer.
        :param wd: Weight decay for the optimizer.
        :param betas: Beta values for the Adam optimizer.
        :param n_optim: Number of parallel processes performing optimization.
        :param gamma: Discount factor for the future rewards.
        :param tau: Soft update coefficient for updating the target network.
        :param total_frames: Total number of frames for training.
        :param init_random_frames: Number of initial random frames for exploration.
        :param frames_per_batch: Number of frames to collect before performing optimization.
        :param batch_size: Batch size used for optimization.
        :param max_buffer_size: Maximum size of the replay buffer.
        :param num_workers: Number of parallel workers for data collection.
        :param num_collectors: Number of parallel workers collecting data from the environment.
        :param eps_greedy_val: Epsilon value for epsilon-greedy exploration strategy.
        :param eps_greedy_val_env: Epsilon value for environment exploration.
        """
        self.env_str = env_str
        self.lr = lr
        self.wd = wd
        self.betas = betas
        self.n_optim = n_optim
        self.gamma = gamma
        self.tau = tau
        self.total_frames = total_frames
        self.init_random_frames = init_random_frames
        self.frames_per_batch = frames_per_batch
        self.batch_size = batch_size

        self.buffer_size = min(max_buffer_size, total_frames)
        self.max_buffer_size = max_buffer_size
        self.num_workers = num_workers
        self.num_collectors = num_collectors
        self.eps_greedy_val = eps_greedy_val
        self.eps_greedy_val_env = eps_greedy_val_env
        self.env_fun = env_fun

        self.log_interval = 500

        self._actor = None
        self._actor_explore = None
        self._target_updater = None
        self._loss_module = None
        self._collector = None
        self._replay_buffer = None
        self._optimizer = None
        self._trainer = None

        # Builder pattern?
        self.init_policy()
        self.init_loss_module()
        self.init_collector()
        self.init_optimizer()
        self.init_trainer()

    def get_policy(self) -> tuple[TensorDictModule, TensorDictSequential]:
        return self._actor, self._actor_explore

    def train(self):
        logger.debug("running trainer...")
        self._trainer.train()

    def eval(self):
        pass

    def init_policy(self):
        """
        :return:
        """
        dummy_env = self.env_fun(self.env_str, 1)

        cnn_kwargs = {
            "num_cells": [32, 64, 64],
            "kernel_sizes": [6, 4, 3],
            "strides": [2, 2, 1],
            "activation_class": nn.ELU,
            # This can be used to reduce the size of the last layer of the CNN
            # "squeeze_output": True,
            # "aggregator_class": nn.AdaptiveAvgPool2d,
            # "aggregator_kwargs": {"output_size": (1, 1)},
        }
        mlp_kwargs = {
            "depth": 2,
            "num_cells": [
                64,
                64,
            ],
            "activation_class": nn.ELU,
        }
        net = DuelingCnnDQNet(
            dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
        ).to(fetch_device())
        # net.value[-1].bias.data.fill_(init_bias)

        actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(fetch_device())
        # init actor: because the model is composed of lazy conv/linear layers,
        # we must pass a fake batch of data through it to instantiate them.
        tensordict = dummy_env.fake_tensordict()
        actor(tensordict)

        # we join our actor with an EGreedyModule for data collection
        exploration_module = EGreedyModule(
            spec=dummy_env.action_spec,
            annealing_num_steps=self.total_frames,
            eps_init=self.eps_greedy_val,
            eps_end=self.eps_greedy_val_env,
        )

        # noinspection PyTypeChecker
        actor_explore = TensorDictSequential(actor, exploration_module)

        self._actor = actor
        self._actor_explore = actor_explore

    def init_loss_module(self):
        loss_module = DQNLoss(self._actor, delay_value=True)
        loss_module.make_value_estimator(gamma=self.gamma)
        target_updater = SoftUpdate(loss_module, eps=0.995)

        self._loss_module = loss_module
        self._target_updater = target_updater

    def init_collector(self):
        self._collector = get_collector(
            self.env_str,
            self.env_fun,
            self.num_workers,
            self.num_collectors,
            self._actor_explore,
            self.frames_per_batch,
            self.total_frames,
            fetch_device(),
            self.gamma,
        )

    def init_replay_buffer(self):
        self._replay_buffer = get_replay_buffer(
            buffer_size=self.buffer_size,
            n_optim=self.n_optim,
            batch_size=self.batch_size
        )

    def init_optimizer(self):
        self._optimizer = Adam(
            self._loss_module.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            betas=self.betas
        )

    def init_trainer(self):
        logger.debug("init trainer....")

        # Temporarely put this here.
        exp_name = f"dqn_exp_{uuid.uuid1()}"
        tmpdir = tempfile.TemporaryDirectory()
        logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
        warnings.warn(f"log dir: {logger.experiment.log_dir}")

        self._trainer = Trainer(
            collector=self._collector,
            total_frames=self.total_frames,
            frame_skip=1,
            loss_module=self._loss_module,
            optimizer=self._optimizer,
            logger=logger,
            optim_steps_per_batch=self.n_optim,
            log_interval=self.log_interval,
        )