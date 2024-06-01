import torch
from botorch.models.gpytorch import GPyTorchModel
from torch.optim import Optimizer


def save_model_and_optimizer(model: GPyTorchModel, optimizer: Optimizer, filepath: str) -> None:
    """
    Save the model state dictionary and optimizer state dictionary to a file.

    :param model: The GPytorch model instance.
    :type model: GPyTorchModel
    :param optimizer: The optimizer used for training the model.
    :type optimizer: Optimizer
    :param filepath: The file path where the state dictionaries will be saved.
    :type filepath: str
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)


def save_model(model: GPyTorchModel, filepath: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict()
    }, filepath)


def save_optimizer(optimizer: Optimizer, filepath: str) -> None:
    torch.save({
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)


def load_model_and_optimizer(model: GPyTorchModel, optimizer: Optimizer, filepath: str) -> None:
    """
    Load the model state dictionary and optimizer state dictionary from a file.

    :param model: The GPytorch model instance.
    :type model: GPyTorchModel
    :param optimizer: The optimizer used for training the model.
    :type optimizer: Optimizer
    :param filepath: The file path from where the state dictionaries will be loaded.
    :type filepath: str
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def load_model(model: GPyTorchModel, filepath: str) -> None:
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])


def load_optimizer(optimizer: Optimizer, filepath: str) -> None:
    checkpoint = torch.load(filepath)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
