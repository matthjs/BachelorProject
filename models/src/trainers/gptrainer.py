from collections import deque

import gpytorch
import torch
from torch import optim

from models.gp import ExactGaussianProcessRegressor


class GaussianProcessTrainer:
    """
    This model is an online trainer.
    """
    def __init__(self, model: ExactGaussianProcessRegressor, learning_rate, max_dataset_size=10000, sparsification=False, optimizer=None):
        self.model = model
        self.mll = gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood(model.likelihood, model)
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=learning_rate)
        self.data_x = deque(maxlen=max_dataset_size)    # Memory intensive: keep track of N latest samples.
        self.data_y = deque(maxlen=max_dataset_size)

    def train(self, train_x, train_y, num_epochs=100, logging=True, hyperparameter_fitting=True) -> None:
        """
        Conditions Gaussian process on new incoming training data (train_x, train_y).
        Optionally also (re)-fits the Gaussian process hyperpameters to (train_x, train_y) data.
        Note that in inference the GP is conditioned on the new data.
        :param train_x: input values compatible with kernel/covariance function.
        :param train_y: scalar target values.
        :param num_epochs: the number of iterations over the train data.
        :param logging:
        :param hyperparameter_fitting:
        """
        print("EPIC!")

        # No linear independence test is performed for now, just add to dataset.
        self.data_x.append(train_x)
        self.data_y.append(train_y)
        self.model.set_train_data(
            torch.cat(list(train_x)),
            torch.cat(list(train_y)).squeeze(),
            strict=False)

        self.model.train()
        self.model.likelihood.train()

        if hyperparameter_fitting:
            # noinspection DuplicatedCode
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                output = self.model(train_x)
                loss = -self.mll(output, train_y)
                loss.backward()
                if logging:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        epoch + 1, num_epochs, loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.item(),
                        self.model.likelihood.noise.item()
                    ))
                self.optimizer.step()

                if logging and epoch % 10 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        # Incorporate new data using fantasy modeling
        # Not now.
        # self.model <- gaussian process conditioned on additional training data.
        # if self.model.train_inputs is not None:
        #    self.model = self.model.get_fantasy_model(train_x, train_y)   # CHECK -> does this update other references?

        self.model.eval()
        self.model.likelihood.eval()
