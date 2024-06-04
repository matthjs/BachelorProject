"""
Based on https://github.com/pytorch/botorch/issues/1750 and
https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html
"""
from typing import Any, Callable, Dict, List, Optional, Type, Union
from math import floor

import gpytorch
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.models.kernels import CategoricalKernel
from botorch.posteriors import GPyTorchPosterior
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

from gp.deepgplayers import DeepGPHiddenLayer, DeepGPMultivariateNormal, DeepGPMixedHiddenLayer


class DeepGPModel(DeepGP, GPyTorchModel):
    def __init__(self,
                 train_x_shape,
                 hidden_layers_config: List[Dict[str, Any]],
                 num_inducing_points=128,
                 cat_dims: Optional[List[int]] = None):
        """
        NOTE: Currently does not allow you to customize the kernel functions used.
        Args:
            train_x_shape: Shape of the training data.
            LAST LAYER SHOULD ALWAYS HAVE MEAN_TYPE "CONSTANT".
            hidden_layers_config: List of dictionaries where each dictionary contains the configuration
                                  for a hidden layer. Each dictionary should have the keys:
                                  - "output_dims": Number of output dimensions.
                                  - "mean_type": Type of mean function ("linear" or "constant").
            num_inducing_points: Number of inducing points for the variational strategy.
        """
        super().__init__()

        input_dims = train_x_shape[-1]
        self.layers = []
        first_layer = True

        # Create hidden layers based on the provided configuration
        for layer_config in hidden_layers_config:
            if first_layer and cat_dims is not None:        # Maybe a bit clunky.
                hidden_layer = DeepGPMixedHiddenLayer(
                    input_dims=input_dims,
                    output_dims=layer_config['output_dims'],
                    mean_type=layer_config['mean_type'],
                    num_inducing=num_inducing_points,
                    cat_dims=cat_dims
                )
                first_layer = False
            else:
                hidden_layer = DeepGPHiddenLayer(
                    input_dims=input_dims,
                    output_dims=layer_config['output_dims'],
                    mean_type=layer_config['mean_type'],
                    num_inducing=num_inducing_points
                )
            self.layers.append(hidden_layer)
            input_dims = layer_config['output_dims']

        # Add all layers as module list
        self.layers = torch.nn.ModuleList(self.layers)
        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
        self.double()
        self.intermediate_outputs = None

    def forward(self, inputs):
        x = inputs
        self.intermediate_outputs = []
        for layer in self.layers:
            x = layer(x)
            self.intermediate_outputs.append(x)
        return x

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

    def get_intermediate_outputs(self):
        return self.intermediate_outputs
