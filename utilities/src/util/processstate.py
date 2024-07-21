import torch

from util.fetchdevice import fetch_device


def process_state(state):
    # StableBaselines vec environments will return [1, state_dim] but we need [state_dim]
    return torch.from_numpy(state).to(device=fetch_device(), dtype=torch.double)