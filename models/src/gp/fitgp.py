import os
from typing import Optional

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL, MarginalLogLikelihood
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader

from util.save import load_model, save_model


class GPFitter:
    """
    Gaussian Process Fitter class to fit a variational GP model.
    """

    def __init__(self):
        """
        Constructor for GPFitter.
        """
        self.first_time = True

    def _construct_mll(self, gp_mode: str, model: GPyTorchModel, train_y: torch.Tensor) -> gpytorch.mlls.MarginalLogLikelihood:
        """
        Construct the marginal log likelihood (MLL) based on the GP mode.

        :param gp_mode: The type of the Gaussian Process ('deep_gp', 'variational_gp', or 'exact_gp'). Formally
        a deep_gp is not a GP, but it shares the same interface.
        :param model: The GPytorch model instance.
        :param train_y: The training output tensor.
        :return: The constructed marginal log likelihood.
        """
        if gp_mode == 'deep_gp':
            mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_y.numel()))
        elif gp_mode == 'variational_gp':
            mll = VariationalELBO(model.likelihood, model.model, train_y.numel())
        else:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll

    def _exact_gp_optimize(self,
                           optimizer: torch.optim.Optimizer,
                           mll: gpytorch.mlls.MarginalLogLikelihood,
                           model: GPyTorchModel,
                           train_x: torch.Tensor,
                           train_y: torch.Tensor,
                           num_epochs: int) -> float:
        """
        Optimize an exact GP model.

        :param optimizer: The optimizer for the model.
        :param mll: The marginal log likelihood.
        :param model: The GPytorch model instance.
        :param train_x: The training input tensor.
        :param train_y: The training output tensor.
        :param num_epochs: The number of training epochs.
        :return: The final loss after optimization.
        """
        loss = 0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)

            if loss < 0:  # Prevents numerical problems.
                break

            loss.backward()
            optimizer.step()

        return loss

    def _variational_gp_optimize(self,
                                 optimizer: torch.optim.Optimizer,
                                 mll: gpytorch.mlls.MarginalLogLikelihood,
                                 model: GPyTorchModel,
                                 train_x: torch.Tensor,
                                 train_y: torch.Tensor,
                                 num_epochs: int,
                                 random_batching: bool,
                                 num_batches: int,
                                 batch_size: int) -> float:
        """
        Optimize a variational GP model.

        :param optimizer: The optimizer for the model.
        :param mll: The marginal log likelihood.
        :param model: The GPytorch model instance.
        :param train_x: The training input tensor.
        :param train_y: The training output tensor.
        :param num_epochs: The number of training epochs.
        :param random_batching: Whether to use random batching.
        :param num_batches: The number of batches.
        :param batch_size: The batch size for mini-batching.
        :return: The total loss after optimization.
        """
        total_loss = 0
        loss = None
        if random_batching:
            train_loader = DataLoader(TensorDataset(train_x, train_y),
                                      batch_size=batch_size,
                                      shuffle=True)
        else:
            data = TensorDataset(train_x, train_y)
            train_loader = DataLoader(data, batch_size=batch_size)

        for x_batch, y_batch in train_loader:
            for _ in range(num_epochs):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            num_batches -= 1

            if num_batches == 0:
                break

        return total_loss

    def __call__(self,
                 model: GPyTorchModel,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 gp_mode: str,
                 batch_size: int = 128,
                 num_epochs: int = 1,
                 num_batches: int = 50,
                 learning_rate: float = 0.001,
                 random_batching: bool = True,
                 logging: bool = False,
                 checkpoint_path: Optional[str] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        """
        Fit a variational Gaussian process model.

        :param model: The GPytorch model instance.
        :param train_x: The training input tensor of shape (batch_size, D).
        :param train_y: The training output tensor of shape (batch_size,).
        :param gp_mode: The mode of the Gaussian Process ('deep_gp', 'variational_gp', or 'exact_gp').
        :param batch_size: The batch size for mini-batching. Defaults to 128.
        :param num_epochs: The number of training epochs. Defaults to 1.
        :param num_batches: The number of batches. Defaults to 50.
        :param learning_rate: The learning rate for the Adam optimizer. Defaults to 0.001.
        :param random_batching: Whether to use random batching. Defaults to True.
        :param logging: Whether to log losses during training. Defaults to False.
        :param checkpoint_path: The file path to save checkpoints during training. If None, checkpoints are not saved. Defaults to None.
        :param optimizer: Optimizer for the model. If None, Adam optimizer is used. Defaults to None.
        :return: The average loss (MLL or ELBO) after training.
        """
        model.train()
        # Conditionally squeeze
        if train_y.dim() > 1 and train_y.shape[1] == 1:
            train_y = train_y.squeeze(1)

        # The model loading is only needed for exact GPs.
        if checkpoint_path and not self.first_time and os.path.exists(checkpoint_path):
            try:
                load_model(model, checkpoint_path)
                logger.info(f"Loaded model from {checkpoint_path}")
            except RuntimeError as err:
                logger.info(f"Failed to load mode")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            if optimizer is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        mll = self._construct_mll(gp_mode, model, train_y)

        total_batches = num_batches
        # Note exact GPs do not support (mini-)batching.
        with gpytorch.settings.debug(True):
            if gp_mode == 'exact_gp':
                total_loss = self._exact_gp_optimize(optimizer, mll, model, train_x, train_y, num_epochs)
            else:
                total_loss = self._variational_gp_optimize(optimizer, mll, model, train_x, train_y,
                                                           num_epochs, random_batching, num_batches, batch_size)

        avg_loss = total_loss / total_batches if total_batches > 0 else total_loss

        if logging:
            logger.debug(f"Average loss (epochs: {num_epochs}) {avg_loss}")
        model.eval()
        if checkpoint_path:
            save_model(model, checkpoint_path)
            logger.info(f"Saved model to {checkpoint_path}")

        self.first_time = False
        return avg_loss
