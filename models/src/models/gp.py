import gpytorch.models


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
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(GaussianProcessRegressor, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
