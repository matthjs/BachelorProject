import torch


def fetch_device() -> str:
    """
    Returns 'cuda' if GPU is available otherwise 'cpu'.
    Can be used to determine where to move tensors to.
    :return: string 'cuda' or 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
