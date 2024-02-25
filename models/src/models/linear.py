import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        out = self._linear(x)
        return out

    def save(self, file_name='./params/model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='./params/model.pth'):
        self.load_state_dict(torch.load(file_name))