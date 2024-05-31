from enum import Enum, auto

import torch
from botorch.models import SingleTaskVariationalGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader


def fit_gp(model: GPyTorchModel,
           train_x: torch.tensor,
           train_y: torch.tensor,  # IMPORTANT, train_y should be (, N) shaped.
           gp_mode: str,
           batch_size=64,  # Probably because train_y.numel() (?)
           num_epochs=100,      # 60
           learning_rate=0.001,
           logging=False
           ) -> None:
    """
    Fit a variational Gaussian processes. Fit variational parameters of
    approximate posterior and inducing points. Uses Adam optimizer for stochastic
    gradient descent.
    :param model: an instance of SingleTaskVariationalGP
    :param train_x: a (batch_size, D) sized tensor.
    :param train_y: a (,D) sized tensor.
        :param gp_mode:
    :param batch_size: mini-batching is used instead of going over entire dataset per epoch.
    :param num_epochs: how many train passes over the dataset.
    :param learning_rate: adam optimizer parameter.
    :param logging: log losses during training or no.
    """
    model.train()
    # Conditionally squeeze
    if train_y.dim() > 1 and train_y.shape[1] == 1:
        print("YO!")
        train_y = train_y.squeeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # The loss of variational GP is an ELBO one like Gaussian NN.
    if gp_mode == 'deep_gp':
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_y.numel()))
    elif gp_mode == 'variational_gp':
        mll = VariationalELBO(model.likelihood, model.model, train_y.numel())
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=batch_size,
                              shuffle=True)

    # TODO: REFACTOR
    # Exact GP Optimization does not allow for mini-batching.
    if gp_mode == 'exact_gp':
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)        # Problem is here.
            loss.backward()
            if epoch % 20 == 0 and logging:
                logger.debug(f"mll loss: {loss}")
            optimizer.step()
            if loss < 0:        # Prevents numerical problems.
                break

    else:
        for _ in range(num_epochs):
            # Within each iteration, we will go over each minibatch of data
            # Batching causes "RuntimeError: You must train on the training inputs!" error with exactGPs.
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)

                loss = -mll(output, y_batch)  # mean() necessary?
                loss.backward()

                if logging:
                    logger.debug(f"variational loss {loss}")

                optimizer.step()

    model.eval()
