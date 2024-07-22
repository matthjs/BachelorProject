"""
Based on https://github.com/pytorch/botorch/issues/1750 and
https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html
"""
from typing import Any, Dict, List, Union, Tuple

import gpytorch
import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor
from torch.utils.data import DataLoader

from gp.deepgplayers import DeepGPHiddenLayer, DeepGPMultivariateNormal


class DeepGPModel(DeepGP, GPyTorchModel):
    """
    Deep Gaussian Process Model class implementing BoTorch and GPyTorch interfaces.
    This class currently does not allow customization of kernel functions and uses RBF for each unit by default.
    """

    def __init__(self,
                 train_x_shape: torch.Size,
                 hidden_layers_config: List[Dict[str, Any]],
                 num_inducing_points: int = 128,
                 input_transform: Any = None,
                 outcome_transform: Any = None):
        """
        Constructor for DeepGPModel.

        :param train_x_shape: Shape of the training data.
        :param hidden_layers_config: List of dictionaries where each dictionary contains the configuration
                                     for a hidden layer. Each dictionary should have the keys:
                                     - "output_dims": Number of output dimensions.
                                     - "mean_type": Type of mean function ("linear" or "constant").
                                     NOTE: The last layer should always have mean_type as "constant".
        :param num_inducing_points: Number of inducing points (per unit) for the variational strategy. Default is 128.
        :param input_transform: Transformation to be applied to the inputs. Default is None.
        :param outcome_transform: Transformation to be applied to the outputs. Default is None.
        """
        super().__init__()

        input_dims = train_x_shape[-1]
        self.layers = []

        # Create hidden layers based on the provided configuration
        for layer_config in hidden_layers_config:
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

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the model.
        Side effect: stores intermediate output representations in a list.

        :param inputs: Input tensor.
        :return: Output tensor after passing through the hidden layers.
        """
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
        """
        Compute the posterior distribution.

        :param X: Input tensor.
        :param observation_noise: Whether to add observation noise. Default is False.
        :return: Posterior distribution.
        """
        self.eval()  # make sure model is in eval mode

        X = self.transform_inputs(X)  # Transform the inputs

        with torch.no_grad() and gpytorch.settings.num_likelihood_samples(10):
            dist = self(X)  # Compute the posterior distribution

            if observation_noise:
                dist = self.likelihood(dist, *args, **kwargs)  # Add observation noise

        posterior = DeepGPMultivariateNormal(distribution=dist)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def predict(self, test_loader: DataLoader) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Make predictions on the test set.
        NOTE: This function should not be used in practice as it does not apply
        any configured input output transformations and only exists
        to ensure compatibility with some Jupyter notebooks.

        :param test_loader: DataLoader for the test set.
        :return: Means, variances, and log likelihoods of the predictions.
        """
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

    def get_intermediate_outputs(self) -> List[Tensor]:
        """
        Get the intermediate outputs from the hidden layers.
        Prerequisite: A forward pass must have been performed.

        :return: List of intermediate outputs.
        """
        return self.intermediate_outputs
