from abc import ABC, abstractmethod

import gpytorch.models
import torch
from loguru import logger
from torch import nn

from util.fetchdevice import fetch_device


class AbstractGPRegressor(ABC):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def set_train_data(self, inputs=None, targets=None, strict=True):
        # Only used by ExactGP models.
        pass


class ExactGaussianProcessRegressorF(gpytorch.models.ExactGP, AbstractGPRegressor):
    def __init__(self,
                 train_x=None,
                 train_y=None,
                 likelihood=gpytorch.likelihoods.GaussianLikelihood().to(device=fetch_device()),
                 mean_function=gpytorch.means.ConstantMean(),
                 covar_function=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())):
        super(ExactGaussianProcessRegressorF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_function
        self.covar_module = covar_function
        self.likelihood = likelihood

    def forward(self, x):
        """
        Give predictions for test inputs.
        :return: a mean output and confidence interval (from bayesian predictive posterior)
        and the multivariate normal after conditioning.
        """
        self.eval()
        self.likelihood.eval()

        print("heyo!")

        # Model predictions are made by feeding the model output through the likelihood.
        # Also WARNINGS are disabled! Done so that we do not get warnings in case we want predictions for
        # training data.
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(state=False):
            f_pred = self.predictive_distribution(x)
            observed_pred = self.likelihood(f_pred)
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
            return mean, lower, upper, f_pred

    def predictive_distribution(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        super(gpytorch.models.ExactGP, self).set_train_data(inputs, targets, strict)


class ExactGaussianProcessRegressor(gpytorch.models.ExactGP, AbstractGPRegressor):
    def __init__(self,
                 train_x=None,
                 train_y=None,
                 likelihood=gpytorch.likelihoods.GaussianLikelihood().to(device=fetch_device()),
                 mean_function=gpytorch.means.ConstantMean(),
                 covar_function=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())):
        super(ExactGaussianProcessRegressor, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_function
        self.covar_module = covar_function
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x) -> tuple:
        """
        Give predictions for test inputs.
        :return: a mean output and confidence interval (from bayesian predictive posterior)
        and the multivariate normal after conditioning.
        """
        self.eval()
        self.likelihood.eval()

        # Model predictions are made by feeding the model output through the likelihood.
        # Also WARNINGS are disabled! Done so that we do not get warnings in case we want predictions for
        # training data.
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(state=False):
            f_pred = self(x)
            observed_pred = self.likelihood(f_pred)
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
            return mean, lower, upper, f_pred

    def set_train_data(self, inputs=None, targets=None, strict=False):
        super(gpytorch.models.ExactGP, self).set_train_data(inputs, targets, strict)


def max_state_action_value(qgp_model: AbstractGPRegressor, action_space_size, state_batch, device=None):
    """
    Helper function for performing the max_a Q(S,a) operation for Q-learning.
    Assumes A is a discrete action space of the form {0, 1, ..., <action_space_size> - 1}.
    Given a batch of states {S_1, ..., S_N} gets {max_a Q(S_1, a), max_a Q(S_2, a), ..., max_a Q(S_n, a)}.
    :param qgp_model: a Gaussian process regression model for Q : S x A -> R.
    :param action_space_size: the number of discrete actions (e.g., env.action_space.n for discrete gym spaces).
    :param state_batch: N state vectors.
    :param device: determines whether new tensors should be placed in main memory (CPU) or VRAM (GPU).
    :return: a vector (PyTorch tensor) of Q-values shape (batch_size, 1).
    """
    with torch.no_grad():
        q_values = []    # for each action, the batched q-values.
        batch_size = state_batch.size(0)
        if device is None:
            device = fetch_device()

        # Assume discrete action encoding starting from 0.
        for action in range(action_space_size):
            action_batch = torch.full((batch_size, 1), action).to(device)
            state_action_pairs = torch.cat((state_batch, action_batch), dim=1).to(device)

            # print(state_action_pairs.shape)

            mean_qs, _, _, _ = qgp_model.predict(state_action_pairs)   # batch_size amount of q_values.
            q_values.append(mean_qs)
            # print(f"S X A: \n{state_action_pairs}, q_values: {mean_qs}\n")

        # Some reshaping black magic to get the max q value along each batch dimension.
        # print(q_values)
        q_tensor = torch.cat(q_values, dim=0).view(len(q_values), -1, 1)
        max_q_values, max_actions = torch.max(q_tensor, dim=0)
        # print(max_q_values)
        # print(max_q_values.shape)
        # print(max_actions)
        # print(max_actions.shape)

    return max_q_values, max_actions


"""
with torch.no_grad():
    q_values = []

    # Again assume discrete action encoding starting from 0.
    for action in range(self.action_space_size):
        print(next_states.size(0))
        action_batch = torch.full((self.batch_size, 1), action).to(self.device)
        next_state_action_pairs = torch.cat((next_states, action_batch), dim=1).to(self.device)
        mean_q, _, _, _ = self.gp.predict(next_state_action_pairs)
        q_values.append(mean_q)

    # Some reshaping black magic to get the max q value along each batch dimension.
    q_tensor = torch.cat(q_values, dim=0).view(len(q_values), -1, 1)
    max_q_values, _ = torch.max(q_tensor, dim=0)
"""

