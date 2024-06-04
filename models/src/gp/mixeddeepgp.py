from typing import Any, Callable, Dict, List, Optional, Type, Union
from math import floor

import gpytorch
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.models.kernels import CategoricalKernel
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels.kernel import Kernel
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor

from gp.deepgplayers import DeepGPMixedHiddenLayer, DeepGPHiddenLayer, DeepGPMultivariateNormal


# noinspection DuplicatedCode
class BotorchDeepGPMixed(DeepGP, GPyTorchModel):
    def __init__(self,
                 train_x_shape,
                 cat_dims: List[int],
                 num_hidden_dims=15,
                 cont_kernel_factory: Optional[
                     Callable[[torch.Size, int, List[int]], Kernel]
                 ] = None,
                 num_inducing_points=128,
                 input_transform=None,
                 outcome_transform=None
                 ):
        hidden_layer = DeepGPMixedHiddenLayer(
            input_dims=train_x_shape[-1],
            cat_dims=cat_dims,
            output_dims=num_hidden_dims,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing=num_inducing_points,
            mean_type="linear",
        )

        hidden_layer2 = DeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=num_inducing_points
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing=num_inducing_points,
            mean_type="constant",
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.hidden_layer2 = hidden_layer2
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
        self.double()

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output

    def posterior(self,
                  X: Tensor,
                  observation_noise: Union[bool, Tensor] = False,
                  *args, **kwargs) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode

        X = self.transform_inputs(X)  # Transform the inputs

        with torch.no_grad():
            dist = self(X)  # Compute the posterior distribution

            if observation_noise:
                dist = self.likelihood(dist, *args, **kwargs)  # Add observation noise

        posterior = DeepGPMultivariateNormal(distribution=dist)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def predict(self, test_loader):
        # Careful this does not transform inputs.
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return (
            torch.cat(mus, dim=-1),
            torch.cat(variances, dim=-1),
            torch.cat(lls, dim=-1),
        )


class BotorchDeepGP(DeepGP, GPyTorchModel):
    def __init__(self, train_x_shape, num_hidden_dims=15, num_inducing_points=128):
        # Apparently num_hidden_dims >= 15
        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=num_inducing_points
        )

        hidden_layer2 = DeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=num_inducing_points
        )

        hidden_layer3 = DeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=num_inducing_points
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type="constant",
            num_inducing=num_inducing_points
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.hidden_layer2 = hidden_layer2
        self.hidden_layer3 = hidden_layer3
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
        self.double()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        hidden_rep3 = self.hidden_layer3(hidden_rep2)
        output = self.last_layer(hidden_rep3)
        return output

    def posterior(self,
                  X: Tensor,
                  observation_noise: Union[bool, Tensor] = False,
                  *args, **kwargs) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode

        X = self.transform_inputs(X)  # Transform the inputs

        with torch.no_grad():
            dist = self(X)  # Compute the posterior distribution
            # print(dist.mean.shape)

            if observation_noise:
                dist = self.likelihood(dist, *args, **kwargs)  # Add observation noise

        posterior = DeepGPMultivariateNormal(distribution=dist)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def predict(self, test_loader):
        # Carefull this does not transform inputs.
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                # print(preds.mean.shape)
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return (
            torch.cat(mus, dim=-1),
            torch.cat(variances, dim=-1),
            torch.cat(lls, dim=-1),
        )
