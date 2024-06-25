import os
from typing import Optional

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL, PredictiveLogLikelihood
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader, Sampler

from util.save import load_model, save_model


# Deprecated.
class ReverseSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        # Return an iterator that yields indices in reverse order
        return iter(range(len(self.data_source) - 1, -1, -1))

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data_source)


class GPFitter:
    def __init__(self):
        self.first_time = True

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
                 optimizer=None) -> float:
        """
        Fit a variational Gaussian process model. This function fits the variational parameters of the
        approximate posterior and inducing points. It uses the Adam optimizer for stochastic gradient descent.

        :param model: The GPytorch model instance.
        :type model: GPyTorchModel
        :param train_x: The training input tensor of shape (batch_size, D).
        :type train_x: Tensor
        :param train_y: The training output tensor of shape (batch_size,).
        :type train_y: Tensor
        :param gp_mode: The mode of the Gaussian Process ('deep_gp', 'variational_gp', or 'exact_gp').
        :type gp_mode: str
        :param batch_size: The batch size for mini-batching. Defaults to 64.
        :type batch_size: int
        :param num_epochs: The number of training epochs. Defaults to 100.
        :type num_epochs: int
        :param learning_rate: The learning rate for the Adam optimizer. Defaults to 0.001.
        :type learning_rate: float
        :param logging: Whether to log losses during training. Defaults to False.
        :type logging: bool
        :param checkpoint_path: The file path to save checkpoints during training. If None, checkpoints are not saved. Defaults to None.
        :type checkpoint_path: Optional[str]
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
                logger.info(f"Failed to load model, if this is for variational GPs then it is because"
                            f"of incompatibility of the shape of the inducing points and variational distribution"
                            f"parameters. (TODO: IS THIS FIXABLE?)")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            if optimizer is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # The loss of variational GP is an ELBO one like Gaussian NN.
        if gp_mode == 'deep_gp':
            mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_y.numel()))
        elif gp_mode == 'variational_gp':
            mll = VariationalELBO(model.likelihood, model.model, train_y.numel())
        else:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        total_loss = 0
        total_batches = 0
        # TODO: REFACTOR
        # Exact GP Optimization does not allow for mini-batching.
        with gpytorch.settings.debug(True):
            if gp_mode == 'exact_gp':
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y)

                    if epoch % 10 == 0 and logging:
                        logger.debug(f"mll loss: {loss}")

                    if loss < 0:  # Prevents numerical problems.
                        break

                    loss.backward()
                    optimizer.step()


            else:
                total_batches = num_batches
                if random_batching:
                    train_loader = DataLoader(TensorDataset(train_x, train_y),
                                              batch_size=batch_size,
                                              shuffle=True)
                else:
                    data = TensorDataset(train_x, train_y)
                    train_loader = DataLoader(data,
                                              batch_size=batch_size)

                # num_batches2 = num_batches
                # Within each iteration, we will go over each minibatch of data
                # Batching causes "RuntimeError: You must train on the training inputs!" error with exactGPs.
                for x_batch, y_batch in train_loader:
                    for _ in range(num_epochs):
                        optimizer.zero_grad()
                        output = model(x_batch)

                        loss = -mll(output, y_batch)
                        # loss_sum += loss.item()
                        loss.backward()

                        optimizer.step()
                    total_loss += loss.item()
                    num_batches -= 1

                    # if num_batches == 0 and logging:
                        # logger.debug(f"variational loss on minibatch (epochs: {num_epochs}) {loss}")

                    if num_batches == 0:
                        break
                    # num_batches = num_batches2

        avg_loss = total_loss / total_batches if total_batches > 0 else loss

        if logging:
            logger.debug(f"Average loss (epochs: {num_epochs}) {avg_loss}")
        model.eval()
        if checkpoint_path:
            save_model(model, checkpoint_path)
            logger.info(f"Saved model to {checkpoint_path}")

        self.first_time = False
        # for param_group in optimizer.param_groups:
        #    print("LR", param_group['lr'])
        return avg_loss
