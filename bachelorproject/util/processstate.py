import numpy as np
import torch
from bachelorproject.util.fetchdevice import fetch_device


def process_state(state):
    if isinstance(state, (int, float, np.integer, np.floating)):
        state = np.array([state])  # Convert scalar to array with one element
    # StableBaselines vec environments will return [1, state_dim] but we need [state_dim]
    return torch.from_numpy(state).to(device=fetch_device(), dtype=torch.double)
