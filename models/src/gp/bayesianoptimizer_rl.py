from collections import deque

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from botorch.models.transforms.outcome import Standardize

from gp.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from util.fetchdevice import fetch_device
import gymnasium as gym


def process_state(state):
    return torch.from_numpy(state).to(device=fetch_device())


def append_actions(state: torch.tensor, action_size: int):
    """
    Append to a state vector (s1, s2, ..., s_n) actions such that we have
    a batch of tensors of the form: (s1, s2, ..., s_n, a_1),
                                    (s1, s2, ..., s_n, a_2),
                                    ...
                                    (s1, s2, ..., s_n, a_m)   m = num of actions.
    :param state:
    :param action_size:
    :return:
    """
    batch_size = state.size(0)

    # Repeat the state vector for each action
    repeated_state = state.repeat_interleave(action_size, dim=0)  # Shape: (batch_size * action_size,
    # num_state_features)

    # Create a tensor for actions ranging from 0 to action_size - 1
    actions = torch.arange(action_size).repeat(batch_size)  # Shape: (batch_size * action_size,)

    # Concatenate the repeated state vectors with the actions
    state_action_pairs = torch.cat([repeated_state, actions.unsqueeze(1)], dim=1)  # Shape: (batch_size *
    # action_size, num_state_features + 1)

    return state_action_pairs


def visualize_distributions(posterior_distribution, action_size):
    """
    Visualize the Gaussian distributions for each action.
    :param posterior_distribution: Posterior distribution of the Q-function GP model.
    :param action_size: The number of possible actions.
    """
    fig, axs = plt.subplots(action_size, 1, figsize=(8, 6), sharex=True)

    for i in range(action_size):
        ax = axs[i]
        mean = posterior_distribution.mean.squeeze()[:, i]
        std = posterior_distribution.variance.squeeze().sqrt()[:,
              i]  # Standard deviation is the square root of variance
        x = torch.arange(len(mean))  # Indices for actions

        # Plot mean and standard deviation as error bars
        ax.errorbar(x, mean.numpy(), yerr=std.numpy(), fmt='o', label=f'Action {i}')
        ax.set_ylabel(f'Action {i} Value')
        ax.set_ylim(mean.min().item() - 1.0, mean.max().item() + 1.0)
        ax.grid(True)

    plt.xlabel('State-Action Pair Index')
    plt.tight_layout()
    plt.show()


def simple_thompson_action_sampler(gpq_model, state_tensor: torch.tensor, action_size: int) -> torch.tensor:
    """
    Thompson sampling for discrete action spaces.
    Assumes last dimension is action dimension.
    :param gpq_model: GP regression model for Q : S x A -> R
    :param state_tensor: a state variable.
    :param action_size: action encoding {0, 1, ..., n - 1}
    :return: best action according to sample q-function
    """
    state_action_pairs = append_actions(state_tensor, action_size)

    posterior_distribution: GPyTorchPosterior = gpq_model.posterior(state_action_pairs)

    sampled_q_values = posterior_distribution.rsample()

    best_action = torch.argmax(sampled_q_values, dim=1)

    return best_action


class BayesianOptimizerRL(AbstractBayesianOptimizerRL):
    """
    Bayesian optimizer.
    To be used in the context of model-free RL
    with discrete action spaces.
    TODO: Look at noise parameter!
    An external optimizer will run the .fit() function. The
    agent will use the choose_next_aaction() for action selection.
    This defines a behavioral policy.
    """

    def __init__(self,
                 model_str: str,
                 max_dataset_size: int,
                 state_size: int,
                 action_space: gym.Space,
                 strategy='thompson_sampling'):
        self.device = fetch_device()
        self._data_x = deque(maxlen=max_dataset_size)  # Memory intensive: keep track of N latest samples.
        self._data_y = deque(maxlen=max_dataset_size)
        self._state_size = state_size
        self._action_size = action_space.n
        self._action_space = action_space
        self._outcome_transform = Standardize(m=1)  # I am guessing m should be 1
        # This Standardization is VERY important as we assume the mean function is 0.
        # If not then we will have problems with the Q values.

        if model_str == 'exact_gp':
            # TODO: Look into variant where we do not instantiate a new GP
            # every time.
            # Then again, maybe it does not really matter.
            self.gp_constructor = MixedSingleTaskGP
            self.likelihood_constructor = GaussianLikelihood
            # Just use the standard Matern kernel for now.
        else:
            raise ValueError(f'Unknown gp model type: {model_str}')

        self._current_gp = self.gp_constructor(train_X=torch.zeros(1, state_size + 1),
                                               train_Y=torch.zeros(1),
                                               cat_dims=[self._state_size + 1],
                                               outcome_transform=self._outcome_transform).to(self.device)

    def fit(self, new_train_x, new_train_y, hyperparameter_fitting=True) -> Model:
        """
        Condition GP on data and fit its hyperparameters.
        :return:
        """
        # TODO: Change out kernel function for something else.
        # TODO: Is it really necessary to instantiate a new GP every time?

        # No linear independence test is performed for now, just add to dataset.
        self.extend_dataset(new_train_x, new_train_y)

        train_x = torch.cat(list(self._data_x))
        train_y = torch.cat(list(self._data_y)).squeeze()

        gp = self.gp_constructor(train_X=train_x,
                                 train_Y=train_y,
                                 cat_dims=[self._state_size + 1],
                                 outcome_transform=self._outcome_transform).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        if hyperparameter_fitting:
            fit_gpytorch_mll(mll)

        self._current_gp = gp
        return gp

    def extend_dataset(self, new_train_x, new_train_y):
        self._data_x.append(new_train_x)
        self._data_y.append(new_train_y)

    def choose_next_action(self, state):
        """
        Choosing the next point works a bit differently here.
        We must fix the state part of the input $z \in S times A$
        and search for the most suitable action part.
        Target is typically TD(0) target.
        :param state:
        :param action:
        :param target:
        :return:
        """
        if self._current_gp is None:
            return self._action_space.sample()

        # Things can be adjusted here later though the magic of *composition*
        return simple_thompson_action_sampler(self._current_gp, state, self._action_size)

    def max_state_action_value(self, state_batch, device=None):
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
            for action in range(self._action_size):
                action_batch = torch.full((batch_size, 1), action).to(device)
                state_action_pairs = torch.cat((state_batch, action_batch), dim=1).to(device)

                # print(state_action_pairs.shape)

                mean_qs = self._current_gp.posterior(state_action_pairs).mean  # batch_size amount of q_values.
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
