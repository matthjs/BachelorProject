import torch
from botorch.models import SingleTaskVariationalGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import VariationalELBO
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader


def fit_variational_gp(model: GPyTorchModel,
                       train_x: torch.tensor,
                       train_y: torch.tensor,  # IMPORTANT, train_y should be (, N) shaped.
                       batch_size=64,  # Probably because train_y.numel() (?)
                       num_epochs=10,
                       learning_rate=0.01,
                       logging=False) -> None:
    """
    Fit a variational Gaussian processes. Fit variational parameters of
    approximate posterior and inducing points. Uses Adam optimizer for stochastic
    gradient descent.
    :param model: an instance of SingleTaskVariationalGP
    :param train_x: a (batch_size, D) sized tensor.
    :param train_y: a (,D) sized tensor.
    :param batch_size: mini-batching is used instead of going over entire dataset per epoch.
    :param num_epochs: how many train passes over the dataset.
    :param learning_rate: adam optimizer parameter.
    :param logging: log losses during training or no.
    """
    model.train()
    train_y = train_y.squeeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # The loss of variational GP is an ELBO one like Gaussian NN.
    mll = VariationalELBO(model.likelihood, model.model, train_y.numel())
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=batch_size,
                              shuffle=True)

    for _ in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)

            loss = -mll(output, y_batch).mean()  # mean() necessary?
            loss.backward()

            if logging:
                logger.debug(f"variational loss {loss}")

            optimizer.step()

    model.eval()

