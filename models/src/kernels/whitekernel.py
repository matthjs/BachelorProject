from typing import Union

import gpytorch
import torch
from linear_operator import LinearOperator
from torch import Tensor
from gpytorch.lazy import DiagLazyVariable, ZeroLazyVariable


class WhiteKernel(gpytorch.kernels.Kernel):
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-05, 100000.0), **kwargs):
        super(WhiteKernel, self).__init__(**kwargs)
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.tensor(noise_level)))
        self.register_constraint("raw_noise", gpytorch.constraints.Interval(*noise_level_bounds))

    """
    Implements (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html)
    k(x1, x2) = noise if x1 == x2 else 0
    """
    def forward(self, x1: Tensor, x2: Tensor, **params) -> Union[Tensor, LinearOperator]:
        self.raw_noise_constraint.transform(self.noise_level)