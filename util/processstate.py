import torch

from util.fetchdevice import fetch_device


def process_state(state):
    return torch.from_numpy(state).to(device=fetch_device(), dtype=torch.double)