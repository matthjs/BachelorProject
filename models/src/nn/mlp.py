import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Multi-layer perceptron model.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Multi-layer perceptron initialization.

        :param input_size: Number of input features.
        :param output_size: Number of output features.
        """
        super(MLP, self).__init__()
        self._layers = nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer perceptron.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        return self._layers(x)

    def save(self, file_name='./model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='./model.pth'):
        self.load_state_dict(torch.load(file_name))
