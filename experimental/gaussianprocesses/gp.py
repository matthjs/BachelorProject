from abc import ABC, abstractmethod

import gpytorch.models
import torch
from torch import nn

from util.fetchdevice import fetch_device

class AbstractGPRegressor(ABC):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predictive_distribution(self, x):
        pass

    @abstractmethod
    def set_train_data(self, inputs=None, targets=None, strict=True):
        # Only used by ExactGP models.
        pass


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