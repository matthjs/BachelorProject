import torch
from botorch.models import SingleTaskGP

from util.fetchdevice import fetch_device


def max_state_action_value(gpq_model: SingleTaskGP, action_space_size, state_batch, device=None):
    """
    Helper function for performing the max_a Q(S,a) operation for Q-learning.
    Assumes A is a discrete action space of the form {0, 1, ..., <action_space_size> - 1}.
    Given a batch of states {S_1, ..., S_N} gets {max_a Q(S_1, a), max_a Q(S_2, a), ..., max_a Q(S_n, a)}.
    :param gpq_model:  a Gaussian process regression model for Q : S x A -> R.
    :param action_space_size: the number of discrete actions (e.g., env.action_space.n for discrete gym spaces).
    :param state_batch: N state vectors.
    :param device: determines whether new tensors should be placed in main memory (CPU) or VRAM (GPU).
    :return: a vector (PyTorch tensor) of Q-values shape (batch_size, 1).
    """
    with torch.no_grad():
        q_values = []  # for each action, the batched q-values.
        batch_size = state_batch.size(0)
        if device is None:
            device = fetch_device()

        # Assume discrete action encoding starting from 0.
        for action in range(action_space_size):
            action_batch = torch.full((batch_size, 1), action).to(device)
            state_action_pairs = torch.cat((state_batch, action_batch), dim=1).to(device)

            # print(state_action_pairs.shape)

            mean_qs = gpq_model.posterior(state_action_pairs).mean  # batch_size amount of q_values.
            q_values.append(mean_qs)
            # print(f"S X A: \n{state_action_pairs}, q_values: {mean_qs}\n")

        # Some reshaping black magic to get the max q value along each batch dimension.
        # print(q_values)
        q_tensor = torch.cat(q_values, dim=0).view(len(q_values), -1, 1)
        max_q_values, max_actions = torch.max(q_tensor, dim=0)
        # print(max_q_values)
        # print(max_q_values.shape)
        # print(max_actions)
        # print(max_actions.shape)

    return max_q_values, max_actions
