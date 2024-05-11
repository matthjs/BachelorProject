from collections import deque

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.input import Normalize, InputStandardize
from loguru import logger
from matplotlib import pyplot as plt
from botorch.models.transforms.outcome import Standardize
from torch import optim

from gp.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from util.fetchdevice import fetch_device
import gymnasium as gym


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


def append_actions(state: torch.tensor, action_size: int, device=None) -> torch.tensor:
    """
    Append to a state vector (s1, s2, ..., s_n) actions such that we have
    a batch of tensors of the form: ((s1, s2, ..., s_n, a_1),
                                     (s1, s2, ..., s_n, a_2),
                                     ...
                                     (s1, s2, ..., s_n, a_m)) where  m = num of actions.
    :param state: state tensor.
    :param action_size: number of actions.
    :param device: GPU or CPU.
    :return: batched state action tensor as described above.
    """
    if device is None:
        device = fetch_device()

    # Repeat the state vector for each action
    repeated_state = torch.stack([state] * action_size)

    # Create a tensor for actions ranging from 0 to action_size - 1
    actions = torch.arange(action_size).to(device)

    # Concatenate the repeated state vectors with the actions
    state_action_pairs = torch.cat([repeated_state, actions.unsqueeze(1)], dim=-1)

    return state_action_pairs


def simple_thompson_action_sampler(gpq_model: GPyTorchModel,
                                   state_tensor: torch.tensor,
                                   action_size: int) -> torch.tensor:
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
    TODO: IDEA: Use target network approach from DQN to improve learning?
    An external optimizer will run the .fit() function. The
    agent will use the choose_next_action() for action selection.
    This defines a behavioral policy.
    WARNING: This class is getting a bit big.
    """

    def __init__(self,
                 model_str: str,
                 random_draws: int,
                 max_dataset_size: int,
                 state_size: int,
                 action_space: gym.Space,
                 strategy='thompson_sampling',
                 sparsfication_treshold=None):
        self.device = fetch_device()
        self._data_x = deque(maxlen=max_dataset_size)  # Memory intensive: keep track of N latest samples.
        self._data_y = deque(maxlen=max_dataset_size)
        self._state_size = state_size
        self._action_size = action_space.n
        self._action_space = action_space

        self._input_transform = None  # InputStandardize(d=state_size + 1)
        self._outcome_transform = Standardize(m=1)  # I am guessing m should be 1
        # This Standardization is VERY important as we assume the mean function is 0.
        # If not then we will have problems with the Q values.
        self._sparsification_treshold = sparsfication_treshold

        if model_str == 'exact_gp':
            # TODO: Look into variant where we do not instantiate a new GP
            # every time.
            # Then again, maybe it does not really matter.
            self.gp_constructor = MixedSingleTaskGP
            self.likelihood_constructor = GaussianLikelihood
            # Just use the standard Matern kernel for now.
        else:
            raise ValueError(f'Unknown gp model type: {model_str}')

        self._current_gp = self.gp_constructor(train_X=torch.zeros(1, state_size + 1, dtype=torch.double),
                                               train_Y=torch.zeros(1, 1, dtype=torch.double),
                                               cat_dims=[self._state_size],
                                               input_transform=self._input_transform,
                                               outcome_transform=self._outcome_transform).to(self.device)

        self._random_draws = random_draws

    def get_current_gp(self):
        return self._current_gp

    def fit(self, new_train_x: torch.tensor, new_train_y: torch.tensor, hyperparameter_fitting=True) -> None:
        """
        Condition GP on data and fit its hyperparameters.
        """
        if self._random_draws > 0:
            return

        # No linear independence test is performed for now, just add to dataset.
        self.extend_dataset(new_train_x, new_train_y)

        train_x, train_y = self.dataset()

        gp = self.gp_constructor(train_X=train_x,
                                 train_Y=train_y,
                                 cat_dims=[self._state_size],
                                 input_transform=self._input_transform,
                                 outcome_transform=self._outcome_transform).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        if hyperparameter_fitting:
            fit_gpytorch_mll(mll)

        self._current_gp = gp

    def _linear_independence_test(self,
                                  test_point: torch.tensor,
                                  data_points: torch.tensor,
                                  kernel_matrix: torch.tensor,
                                  kernel_fun) -> bool:
        """
        Computes whether the linear independence test is passed.
        The linear independence test asks, how well is test_point approximated by the elements
        of data_points?
        $test > treshold$?
        $ test = k(z', z') - k(Z, z')^T K(Z, Z)^{-1} k(Z, z')$
        :param test_point: a (1, vec_dim) vector.
        :param data_points: the input dataset (dataset_size, vec_dim).
        :param kernel_matrix: pre-computed from dataset (dataset_size, dataset_size).
        :param kernel_fun: the covariance function.
        :return: whether the independence test is passed depending on the sparsification treshold.
        """
        kernel_vec = kernel_fun(test_point, data_points)
        res = torch.linalg.solve(kernel_matrix, kernel_vec.squeeze(0)).unsqueeze(0)
        res2 = torch.matmul(res, kernel_vec.t())
        test_val = (kernel_fun(test_point, test_point) - res2).to_dense()

        print(f"test result test_val > spars_tresh -> {test_val.item()} > {self._sparsification_treshold}")
        return test_val.item() > self._sparsification_treshold

    def _sparse_add(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        covar_fun = self._current_gp.covar_module
        train_x = torch.cat(list(self._data_x))
        covar_matrix = covar_fun(train_x, train_x)

        # Problem: this is a sequential
        candidates_x = []
        candidates_y = []
        for x, y in zip(new_train_x, new_train_y):
            if self._linear_independence_test(x.unsqueeze(0), train_x, covar_matrix, covar_fun):
                candidates_x.append(x)
                candidates_y.append(y)

        if not candidates_x:
            return

        candidates_x_tensor = torch.cat(candidates_x)
        candidates_y_tensor = torch.cat(candidates_y)

        print("TENSOR SHAPE ->", candidates_x_tensor.shape)

        self._data_x.append(candidates_x_tensor)
        self._data_y.append(candidates_y_tensor)

    def extend_dataset(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        # dequeue evaluates to False if empty.
        if self._sparsification_treshold is not None and self._data_x:
            self._sparse_add(new_train_x, new_train_y)
            return

        self._data_x.append(new_train_x)
        self._data_y.append(new_train_y)

    def dataset(self) -> tuple[torch.tensor, torch.tensor]:
        train_x = torch.cat(list(self._data_x))
        train_y = torch.cat(list(self._data_y))
        return train_x, train_y

    def choose_next_action(self, state: torch.tensor) -> int:
        """
        Choose the next action.

        Choosing the next point works a bit differently compared to traditional BayesOpt.
        We must fix the state part of the input :math: `z \in S times A`
        and search for the most suitable action part.
        Target is typically TD(0) target.
        :param state: a state tensor.
        :return: an action encoded as an int.
        """
        if self._random_draws > 0:
            self._random_draws -= 1
            return self._action_space.sample()

        # Things can be adjusted here later though the magic of *composition*
        action_tensor = simple_thompson_action_sampler(self._current_gp, state, self._action_size)
        return action_tensor.item()

    def state_action_value(self, state_batch: torch.tensor, action_batch: torch.tensor) -> torch.tensor:
        """
        Get the Q-value for batched state action pairs.
        The Q-value taken is the (posterior) mean function of the GP model.
        :param state_batch: a (batch_size, state_dim) state feature tensor.
        :param action_batch: a (batch_size, 1) action tensor. The class assumes discrete action selection.
        :return: a (batch_size, 1) tensor of q values.
        """
        with torch.no_grad():
            next_state_action_pairs = torch.cat((state_batch, action_batch), dim=1).to(self.device)
            q_val = self._current_gp.posterior(next_state_action_pairs).mean

        return q_val

    def max_state_action_value(self, state_batch: torch.tensor, device=None) -> tuple[torch.tensor, torch.tensor]:
        """
        Helper function for performing the max_a Q(S,a) operation for Q-learning.
        Assumes A is a discrete action space of the form {0, 1, ..., <action_space_size> - 1}.
        Given a batch of states {S_1, ..., S_N} gets {max_a Q(S_1, a), max_a Q(S_2, a), ..., max_a Q(S_n, a)}.
        :param state_batch: N state vectors.
        :param device: determines whether new tensors should be placed in main memory (CPU) or VRAM (GPU).
        :return: a vector (PyTorch tensor) of Q-values shape (batch_size, 1) and actions.
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
            # print("maxq->", max_q_values)
            # print(max_q_values.shape)
            # print("maxaction->", max_actions)
            # print(max_actions.shape)

        # print(max_actions.shape)

        return max_q_values, max_actions
