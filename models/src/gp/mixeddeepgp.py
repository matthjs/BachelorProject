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

from gp.deepgp import DeepGPMultivariateNormal


class DeepGPMixedHiddenLayer(DeepGPLayer):
    """
    A Deep GP layer that supports mixed (continuous and categorical) input spaces.
    """

    def __init__(self,
                 input_dims: int,
                 cat_dims: List[int],
                 output_dims: Optional[int] = None,
                 num_inducing: int = 128,
                 mean_type: str = "constant",
                 cont_kernel_factory: Optional[
                     Callable[[torch.Size, int, List[int]], Kernel]
                 ] = None):
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

        super(DeepGPMixedHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims
        )

        if mean_type == "constant":
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        d = input_dims
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        if cont_kernel_factory is None:
            def cont_kernel_factory(
                    batch_shape: torch.Size,
                    ard_num_dims: int,
                    active_dims: List[int],
            ) -> MaternKernel:
                return MaternKernel(
                    nu=2.5,
                    batch_shape=batch_shape,
                    ard_num_dims=ard_num_dims,
                    active_dims=active_dims,
                    lengthscale_constraint=GreaterThan(1e-04),
                )

        if len(ord_dims) == 0:
            self.covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
        else:
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                    )
                )
            )
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
            self.covar_module = sum_kernel + prod_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ to add concatenation based skip connections.
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


# noinspection DuplicatedCode
class BotorchDeepGPMixed(DeepGP, GPyTorchModel):
    def __init__(self,
                 train_x_shape,
                 cat_dims: List[int],
                 num_hidden_dims=10,
                 cont_kernel_factory: Optional[
                     Callable[[torch.Size, int, List[int]], Kernel]
                 ] = None
                 ):
        hidden_layer = DeepGPMixedHiddenLayer(
            input_dims=train_x_shape[-1],
            cat_dims=cat_dims,
            output_dims=num_hidden_dims,
            cont_kernel_factory=cont_kernel_factory,
            mean_type="linear",
        )

        last_layer = DeepGPMixedHiddenLayer(
            input_dims=hidden_layer.output_dims,
            cat_dims=cat_dims,
            output_dims=None,
            cont_kernel_factory=cont_kernel_factory,
            mean_type="constant",
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
        self.double()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
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
