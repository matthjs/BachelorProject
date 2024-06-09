import torch
import torch.nn as nn


# NOT USED!


class LinearModel(nn.Module):
    """
    Linear model (y_hat = xw), corresponding to a single layer multi-layer
    perceptron with the identify function as activation function.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Linear model initialization.

        :param input_size: Number of input features.
        :param output_size: Number of output features.
        """
        super(LinearModel, self).__init__()
        self._linear = nn.Linear(input_size, output_size)
        self.double()

    def _normalize_to_unit_cube(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor to the unit cube [0, 1].

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Normalized tensor of shape (batch_size, input_size).
        """
        min_vals = torch.min(x, dim=0, keepdim=True).values
        max_vals = torch.max(x, dim=0, keepdim=True).values
        normalized_x = (x - min_vals) / (max_vals - min_vals)
        return normalized_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        x = self._normalize_to_unit_cube(x)
        out = self._linear(x)
        return out

    def save(self, file_name='./params/model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='./params/model.pth'):
        self.load_state_dict(torch.load(file_name))