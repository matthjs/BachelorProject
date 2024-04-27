import torch
import torch.nn as nn

from models.gp import ExactGaussianProcessRegressor
from models.linear import LinearModel
from models.mlp import MLP


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, input_size: int, output_size: int) -> nn.Module:
        """
        Create a machine learning model based on the specified type.

        :param device: CPU or GPU.
        :param model_type: Type of model to create.
        :param input_size: Number of input features.
        :param output_size: Number of output features.
        :return: Model instance.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_type == "linear":
            return LinearModel(input_size, output_size).to(device)
        elif model_type == "mlp":
            return MLP(input_size, output_size).to(device)
        elif model_type == "exact_gp":
            return ExactGaussianProcessRegressor().to(device)
        else:
            raise ValueError("Invalid model type. Supported types: 'linear', 'mlp'.")