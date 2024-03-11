import gpytorch.models
import torch

from util.fetchdevice import fetch_device


class DynamicsGPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super(DynamicsGPModel, self).__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood().to(device=fetch_device())):
        super(GaussianProcessRegressor, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
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
