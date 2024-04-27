import tempfile
import uuid
import warnings

import hydra
import torch
from loguru import logger
from tensordict.nn import TensorDictSequential, TensorDictModule
from torch import nn
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec
from torchrl.modules import QValueActor, QValueModule, EGreedyModule, DuelingCnnDQNet, MLP
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record import CSVLogger
from torchrl.trainers import Trainer

from experimental.common.fetchdevice import fetch_device
from experimental.common.rlutil import atari_preprocessing, get_collector, get_replay_buffer, make_env


class DQNAgent:
    @hydra.main(config_path="", config_name="config_dqn_mountaincar", version_base="1.1")
    def __init__(self, cfg: "DictConfig"):   # noqa: F821
        """
        Class for creating components of DQN agent.
        """
        self.env_name = cfg.env.env_name

        self._actor = None
        self._actor_explore = None

        # Builder pattern?
        self.init_policy(cfg.env.env_name)
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

    def init_policy(self, env_name):
        """
        :return:
        """
        dummy_env = make_env(self.env_str)

        # Define input shape
        input_shape = dummy_env.observation_spec["observation"].shape
        env_specs = dummy_env.specs
        num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
        action_spec = env_specs["input_spec", "full_action_spec", "action"]

        # Define Q-Value Module
        mlp = MLP(
            in_features=input_shape[-1],
            activation_class=torch.nn.ReLU,
            out_features=num_outputs,
            num_cells=[120, 84],
        )

        actor = QValueActor(
            module=mlp,
            spec=CompositeSpec(action=action_spec),
            in_keys=["observation"],
        ).to(fetch_device())

        # we join our actor with an EGreedyModule for data collection
        exploration_module = EGreedyModule(
            spec=dummy_env.action_spec,
            annealing_num_steps=self.total_frames,
            eps_init=self.eps_greedy_val,
            eps_end=self.eps_greedy_val_env,
        )

        # noinspection PyTypeChecker
        actor_explore = TensorDictSequential(actor, exploration_module).to(fetch_device())

        del dummy_env

        self._actor = actor
        self._actor_explore = actor_explore

    def init_loss_module(self):
        loss_module = DQNLoss(self._actor, delay_value=True)
        loss_module.make_value_estimator(gamma=self.gamma)
        target_updater = SoftUpdate(loss_module, eps=0.995)

        self._loss_module = loss_module
        self._target_updater = target_updater

    def init_collector(self):
        self._collector = SyncDataCollector(
            create_env_fn=make_env(cfg.env.env_name, "cpu"),
            policy=model_explore,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            device="cpu",
            storing_device="cpu",
            max_frames_per_traj=-1,
            init_random_frames=cfg.collector.init_random_frames,
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