from collections import deque

import torch
from botorch import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from loguru import logger

from bayesopt.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from bayesopt.acquisition import simple_thompson_action_sampler, upper_confidence_bound_selector, ThompsonSampling, \
    UpperConfidenceBound, GPEpsilonGreedy
from gp.fitvariationalgp import fit_variational_gp
from gp.gpviz import plot_gp_point_distribution, plot_gp_contours_with_uncertainty, plot_gp_surface_with_uncertainty
from gp.gpviz2 import plot_gp_contours_with_uncertainty2
from gp.variationalgp import MixedSingleTaskVariationalGP
from kernels.kernelfactory import create_kernel
from util.fetchdevice import fetch_device
import gymnasium as gym


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
                 kernel_type,
                 kernel_args,
                 strategy='upper_confidence_bound',
                 sparsfication_treshold=None,
                 state_space=None):
        self.device = fetch_device()
        self._data_x = deque(maxlen=max_dataset_size)  # Memory intensive: keep track of N latest samples.
        self._data_y = deque(maxlen=max_dataset_size)
        self._state_size = state_size
        self._action_size = action_space.n
        self._action_space = action_space
        self._state_space = state_space

        self._input_transform = Normalize(d=state_size + 1)
        self._outcome_transform = Standardize(m=1)  # I am guessing m should be 1
        # This Standardization is VERY important as we assume the mean function is 0.
        # If not then we will have problems with the Q values.
        self._sparsification_treshold = sparsfication_treshold

        self._kernel_factory = create_kernel(kernel_type, kernel_args)

        if model_str not in ['exact_gp', 'variational_gp']:
            raise ValueError(f'Unknown gp model type: {model_str}')

        self._gp_mode = model_str

        self._current_gp = None
        self._current_gp = self._construct_gp(torch.zeros(10, state_size + 1, dtype=torch.double),
                                              torch.zeros(10, 1, dtype=torch.double))

        self._random_draws = random_draws
        self._dummy_counter = 0
        self._visualize = False

        self._gp_action_selector = None

        if strategy == 'thompson_sampling':
            self._gp_action_selector = ThompsonSampling(action_size=self._action_size)
        elif strategy == 'upper_confidence_bound':
            self._gp_action_selector = UpperConfidenceBound(action_size=self._action_size)
        elif strategy == 'GPEpsilonGreedy':
            self._gp_action_selector = GPEpsilonGreedy(action_space=action_space)

    def _construct_gp(self, train_x, train_y) -> GPyTorchModel:
        del self._current_gp

        if self._gp_mode == 'exact_gp':
            return MixedSingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                cat_dims=[self._state_size],
                cont_kernel_factory=self._kernel_factory,
                input_transform=Normalize(
                    d=self._state_size + 1,
                    indices=list(range(self._state_size))),     # ONLY normalize state part.
                outcome_transform=None
            ).to(self.device)
        elif self._gp_mode == 'variational_gp':
            # Do not use RFF with variational GP.
            return MixedSingleTaskVariationalGP(
                train_X=train_x,
                train_Y=train_y,
                cat_dims=[self._state_size],
                cont_kernel_factory=self._kernel_factory,
                inducing_points=128,     # TODO, make this configurable,
                input_transform=Normalize(d=self._state_size + 1),
                outcome_transform=None
            ).to(self.device)

    def get_current_gp(self):
        return self._current_gp

    def fit(self, new_train_x: torch.tensor, new_train_y: torch.tensor, hyperparameter_fitting=True) -> None:
        """
        Condition GP on data and fit its hyperparameters.
        """
        if self._random_draws > 0:
            return

        self._dummy_counter += 1

        self.extend_dataset(new_train_x,
                            new_train_y)

        train_x, train_y = self.dataset()

        # print(train_x.shape)
        # print(train_y.shape)
        # print(train_y.squeeze(1).shape)

        gp = self._construct_gp(train_x, train_y)

        if hyperparameter_fitting:
            if self._gp_mode == 'exact_gp':
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
            elif self._gp_mode == 'variational_gp':
                fit_variational_gp(gp, train_x, train_y)

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

        # print("kernel val", kernel_fun(test_point, test_point).to_dense())

        print(f"test result test_val > spars_tresh -> {test_val.item()} > {self._sparsification_treshold}")
        return test_val.item() > self._sparsification_treshold

    def _sparse_add(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        covar_fun = self._current_gp.covar_module
        train_x = torch.cat(list(self._data_x))
        covar_matrix = covar_fun(train_x, train_x)

        for x, y in zip(new_train_x, new_train_y):
            if self._linear_independence_test(x.unsqueeze(0), train_x, covar_matrix, covar_fun):
                self._data_x.append(x.unsqueeze(0))
                self._data_y.append(y.unsqueeze(0))

    def extend_dataset(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        # dequeue evaluates to False if empty.
        if self._sparsification_treshold is not None and self._data_x:
            self._sparse_add(new_train_x, new_train_y)
            return

        # Maybe there is a better way to do this
        for x, y in zip(new_train_x, new_train_y):
            self._data_x.append(x.unsqueeze(0))
            self._data_y.append(y.unsqueeze(0))

    def dataset(self) -> tuple[torch.tensor, torch.tensor]:
        train_x = torch.cat(list(self._data_x))
        train_y = torch.cat(list(self._data_y))

        # train_x -> (dataset_size, input_dim) -> OK
        # train_y -> (dataset_size, 1) -> NOT OK => (, dataset_size)

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

        action_tensor = self._gp_action_selector.action(self._current_gp, state)

        if self._dummy_counter == 3:
            """
            plot_gp_contours_with_uncertainty2(self._current_gp,
                                               4,
                                               self._action_size,
                                               (2, 3),
                                               highlight_point=(state[2], state[3], action_tensor.item()),
                                               title='Action-value GP contour'
                                               )

            plot_gp_surface_with_uncertainty(self._current_gp,
                                             (0, 1),
                                             (0, 1),
                                             self._action_size,
                                             highlight_point=(state[2], state[3], action_tensor.item()),
                                             title='Action-value GP'
                                             )
            logger.debug(f"Dataset size {len(self._data_x)}")
            """
            plot_gp_point_distribution(self._current_gp,
                                       state,
                                       self._action_size,
                                       title=f'Point Distribution for state ({state})')
            self._dummy_counter = 0

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
