"""
This module is probably not necessary anymore, since botorch GPs are also GPytorch GPs.
"""
from botorch.models import SingleTaskVariationalGP
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import copy
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors import GammaPrior
from gpytorch.variational import _VariationalDistribution, _VariationalStrategy, VariationalStrategy
from torch import Tensor


class ExactGaussianProcessRegressor:
    pass


class MixedSingleTaskVariationalGP(SingleTaskVariationalGP):
    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            cat_dims: List[int],
            cont_kernel_factory: Optional[
                Callable[[torch.Size, int, List[int]], Kernel]
            ] = None,
            likelihood: Optional[Likelihood] = None,
            outcome_transform: Optional[OutcomeTransform] = None,  # TODO
            input_transform: Optional[InputTransform] = None,  # TODO,
            # variational parameters
            num_outputs: int = 1,
            learn_inducing_points: bool = True,
            variational_distribution: Optional[_VariationalDistribution] = None,
            variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
            inducing_points: Optional[Union[Tensor, int]] = None,
            inducing_point_allocator: Optional[InducingPointAllocator] = None
    ) -> None:
        r"""A single-task variational GP model following [hensman2013svgp] supporting categorical parameters.

        By default, the inducing points are initialized though the
        `GreedyVarianceReduction` of [burt2020svgp]_, which is known to be
        effective for building globally accurate models. However, custom
        inducing point allocators designed for specific down-stream tasks can also be
        provided (see [moss2023ipa]_ for details), e.g. `GreedyImprovementReduction`
        when the goal is to build a model suitable for standard BO.

        A single-task variational GP using relatively strong priors on the Kernel
        hyperparameters, which work best when covariates are normalized to the unit
        cube and outcomes are standardized (zero mean, unit variance).

        This model works in batch mode (each batch having its own hyperparameters).
        When the training observations include multiple outputs, this model will use
        batching to model outputs independently. However, batches of multi-output models
        are not supported at this time, if you need to use those, please use a
        ModelListGP.

        Use this model if you have a lot of data or if your responses are non-Gaussian.

        To train this model, you should use gpytorch.mlls.VariationalELBO and not
        the exact marginal log likelihood.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            cat_dims: A list of indices corresponding to the columns of
                the input `X` that should be considered categorical features.
            cont_kernel_factory: A method that accepts  `batch_shape`, `ard_num_dims`,
                and `active_dims` arguments and returns an instantiated GPyTorch
                `Kernel` object to be used as the base kernel for the continuous
                dimensions. If omitted, this model uses a Matern-2.5 kernel as
                the kernel for the ordinal parameters.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass. Only input transforms are allowed which do not
                transform the categorical dimensions. If you want to use it
                for example in combination with a `OneHotToNumeric` input transform
                one has to instantiate the transform with `transform_on_train` == False
                and pass in the already transformed input.
        """
        if len(cat_dims) == 0:
            raise ValueError(
                "Must specify categorical dimensions for MixedSingleTaskGP"
            )
        self._ignore_X_dims_scaling_check = cat_dims
        input_batch_shape = train_X.shape[:-2]
        print("check", input_batch_shape)
        aug_batch_shape = copy.deepcopy(input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size((num_outputs,))
        self._aug_batch_shape = aug_batch_shape

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

        if likelihood is None:
            # This Gamma prior is quite close to the Horseshoe prior
            min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
            likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_noise, transform=None, initial_value=1e-3
                ),
                noise_prior=GammaPrior(0.9, 10.0),
            )

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))
        if len(ord_dims) == 0:
            covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
        else:
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                    )
                )
            )
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
            covar_module = sum_kernel + prod_kernel
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            inducing_point_allocator=inducing_point_allocator
        )
