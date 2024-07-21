from typing import Optional

import gpytorch
import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor


class DeepGPMultivariateNormal(GPyTorchPosterior):
    def rsample(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        return super().rsample().mean(1)

    @property
    def mean(self) -> Tensor:
        return super().mean.mean(0)  # Is this correct?

    @property
    def variance(self) -> Tensor:
        return super().variance.mean(0)


class DeepGPHiddenLayer(DeepGPLayer):
    """
    This is essentially a GP using variational inference.
    """

    def __init__(self,
                 input_dims,
                 output_dims,
                 num_inducing=128,
                 mean_type="constant"):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super(DeepGPHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims
        )

        if mean_type == "constant":
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(
                    gpytorch.settings.num_likelihood_samples.value(), *inp.shape
                )
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))
