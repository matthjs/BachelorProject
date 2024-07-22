from gpytorch.kernels import RFFKernel, MaternKernel, RBFKernel, SpectralMixtureKernel, Kernel
import torch
from typing import List, Tuple, Callable


class RFFKernelAdapter:
    """
    Adapter class for RFFKernel to conform to the kernel creation interface.

    Problem with MixedSingleTaskGP is that it strictly allows kernels to only specify
    the batch_shape, ard_num_dims, and lengthscale constraint.
    """
    num_samples = 1024

    @staticmethod
    def create_kernel(batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]) -> RFFKernel:
        """
        Create an RFFKernel with specified parameters.

        :param batch_shape: Shape of the batch.
        :param ard_num_dims: Number of dimensions for automatic relevance determination.
        :param active_dims: List of active dimensions.
        :return: RFFKernel instance.
        """
        return RFFKernel(
            num_samples=RFFKernelAdapter.num_samples,
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims
        )


class SpectralMixtureAdapter:
    """
    Adapter class for SpectralMixtureKernel to conform to the kernel creation interface.
    """
    num_mixtures = 4

    @staticmethod
    def create_kernel(batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]) -> SpectralMixtureKernel:
        """
        Create a SpectralMixtureKernel with specified parameters.

        :param batch_shape: Shape of the batch.
        :param ard_num_dims: Number of dimensions for automatic relevance determination.
        :param active_dims: List of active dimensions.
        :return: SpectralMixtureKernel instance.
        """
        return SpectralMixtureKernel(
            num_mixtures=SpectralMixtureAdapter.num_mixtures,
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
            eps=1e-2
        )


def create_kernel(kernel_name: str, cfg) -> Tuple[Callable, bool]:
    """
    Create a kernel based on the given name and configuration.
    Default is RBF kernel.

    Note: This returns the constructor of the kernel, not the kernel itself.
    MixedSingleTaskGP should already put a ScaleKernel on the kernel.

    :param kernel_name: Name of the kernel to create.
    :param cfg: Configuration object with kernel parameters.
    :return: Tuple containing the kernel constructor and a boolean indicating if the kernel uses a scale kernel.
    """
    if kernel_name == "matern":
        return MaternKernel, True
    elif kernel_name == "rff":
        RFFKernelAdapter.num_samples = cfg.rff_kernel.num_samples
        return RFFKernelAdapter.create_kernel, True
    elif kernel_name == "spectral_mixture":
        return SpectralMixtureAdapter.create_kernel, False

    return RBFKernel, True
