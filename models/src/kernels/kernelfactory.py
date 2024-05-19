from gpytorch.kernels import Kernel, RFFKernel, MaternKernel, RBFKernel
import torch
from typing import Any, Callable, Dict, List, Optional


class RFFKernelAdapter:
    """
    Problem with MixedSigleTaskGP is that it strictly allows kernels to only specify
    the batch_shape, ard_num_dims and lengthscale contraint.
    """
    num_samples = 1024

    @staticmethod
    def create_kernel(batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]) -> RFFKernel:
        return RFFKernel(
            num_samples=RFFKernelAdapter.num_samples,
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims
        )


def create_kernel(kernel_name: str, cfg):
    """
    Not returns the constructor of the kernel not the kernel itself.
    MixedSingleTaskGP should already put a ScaleKernel on the kernel.
    :param kernel_name:
    :param cfg:
    :return:
    """
    if kernel_name == "matern":
        return MaternKernel
    elif kernel_name == "rbf":
        return RBFKernel
    elif kernel_name == "rff":
        RFFKernelAdapter.num_samples = cfg.rff_kernel.num_samples
        return RFFKernelAdapter.create_kernel

    return None
