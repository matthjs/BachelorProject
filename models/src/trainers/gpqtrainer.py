import torch
from torchrl.data import ReplayBuffer

from models.gp import GaussianProcessRegressor
from trainers.gptrainer import GaussianProcessTrainer
from trainers.trainer import Trainer


class ExactGPQTrainer(Trainer):
    def __init__(self,
                 model: GaussianProcessRegressor,
                 batch_size: int,
                 buf: ReplayBuffer,
                 learning_rate: float,
                 discount_factor: float,
                 num_epochs=100):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buf = buf

        self.gp = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.gp_trainer = GaussianProcessTrainer(
            self.gp,
            learning_rate=learning_rate,
        )

        self.num_epochs = num_epochs

    def train(self) -> None:
        pass