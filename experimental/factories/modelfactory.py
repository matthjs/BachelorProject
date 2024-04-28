import torch
import torch.nn as nn

from experimental.gaussianprocesses.gp import ExactGaussianProcessRegressor


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
            pass
            # return LinearModel(input_size, output_size).to(device)
        elif model_type == "mlp":
            pass
            # return MLP(input_size, output_size).to(device)
        elif model_type == "exact_gp_value":
            return ExactGaussianProcessRegressor()
        else:
            raise ValueError("Invalid model type. Supported types: 'linear', 'mlp'.")