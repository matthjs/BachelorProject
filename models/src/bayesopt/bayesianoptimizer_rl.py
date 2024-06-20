from collections import deque

import torch
import time
import botorch.settings
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import Normalize
from gpytorch.means import ConstantMean
from loguru import logger

from bachelorproject.configobject import Config
from bayesopt.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from bayesopt.acquisition import ThompsonSampling, \
    UpperConfidenceBound, GPEpsilonGreedy
from gp.custommixedgp import MixedSingleTaskGP
from gp.deepgp import DeepGPModel
from gp.fitgp import GPFitter
from gp.gpviz import plot_gp_point_distribution
from gp.mixeddeepgp import BotorchDeepGPMixed
from gp.mixedvariationalgp import MixedSingleTaskVariationalGP
from kernels.kernelfactory import create_kernel
from util.fetchdevice import fetch_device
import gymnasium as gym


class BayesianOptimizerRL(AbstractBayesianOptimizerRL):
    """
    Bayesian optimizer for model-free RL with discrete action spaces.
    WARNING: This class is getting a bit big.
    """

    def __init__(self,
                 model_str: str,
                 exploring_starts: int,
                 max_dataset_size: int,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 kernel_type: str,
                 kernel_args,
                 strategy='GPEpsilonGreedy',
                 sparsification_treshold: float = None,
                 posterior_observation_noise: bool = False,
                 num_inducing_points: int = 64
                 ):
        """
        Constructor.

        :param model_str: What GP model to use ('exact_gp', 'variational_gp', 'deep_gp').
        :param exploring_starts: Number of choose_next_action() calls before the GP gets updated.
        :param max_dataset_size: Max dataset size to apply.
        :param state_space: The state space of the environment.
        :param action_space: The action space of the environment (has to be discrete).
        :param kernel_type: Type of kernel to be used for states ('matern', 'rff', 'rbf').
        :param kernel_args: Only applicable for RFF for specifying number of samples (is a data object).
        :param strategy: Acquisition function/behavioral policy to use.
        :param sparsification_treshold: If set, attempts to run a sparsification algorithm for the dataset.
        """
        self.device = fetch_device()
        self._data_x = deque(maxlen=max_dataset_size)  # Memory consideration: keep track of N latest samples.
        self._data_y = deque(maxlen=max_dataset_size)
        self._posterior_obs_noise = posterior_observation_noise
        self._num_inducing_points = num_inducing_points

        state_size = state_space.shape[0]
        self._state_size = state_size
        self._action_size = action_space.n
        self._action_space = action_space
        self._state_space = state_space
        self._exploring_starts = exploring_starts
        self._dummy_counter = 0
        self._visualize = False
        self._stupid_flag_that_should_be_removed = True

        self._sparsification_treshold = sparsification_treshold

        self._kernel_factory, self.use_scale_kernel = create_kernel(kernel_type, kernel_args)

        if model_str not in ['exact_gp', 'variational_gp', 'deep_gp']:
            raise ValueError(f'Unknown gp model type: {model_str}')

        self.fit_gp = GPFitter()
        self.latest_loss = 0

        # Initialize GP settings. The GP is filled with a few dummy values.
        self._gp_mode = model_str
        self._current_gp = None
        self._current_gp = self._construct_gp(torch.zeros(10, state_size + 1, dtype=torch.double),
                                              torch.zeros(10, 1, dtype=torch.double), first_time=True)
        self._optimizer = None

        # Initialize actions selector.
        self._gp_action_selector = None
        if strategy == 'thompson_sampling':
            self._gp_action_selector = ThompsonSampling(action_size=self._action_size,
                                                        observation_noise=self._posterior_obs_noise)
        elif strategy == 'upper_confidence_bound':
            self._gp_action_selector = UpperConfidenceBound(action_size=self._action_size,
                                                            beta=Config.UCB_BETA,
                                                            observation_noise=self._posterior_obs_noise)
        elif strategy == 'epsilon_greedy':
            self._gp_action_selector = GPEpsilonGreedy(action_space=action_space,
                                                       annealing_num_steps=Config.GP_E_GREEDY_STEPS,
                                                       observation_noise=self._posterior_obs_noise)

    def _construct_gp(self, train_x, train_y, first_time=False) -> GPyTorchModel:
        """
        Construct a Gaussian process with data (train_x, train_y).

        :param train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param train_y: A tensor with expected shape (dataset_size, 1).
        :return: A GP conditioned on (train_x, train_y) (Hyperparameters NOT fitted).
        """
        if self._gp_mode not in ["variational_gp", "deep_gp"]:
            del self._current_gp

        # mean_module = ConstantMean(batch_shape=train_x.shape[:-2])
        # mean_module.initialize(constant=1)

        if self._gp_mode == 'exact_gp':
            return MixedSingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                # mean_module=mean_module,
                cat_dims=[self._state_size],
                cont_kernel_factory=self._kernel_factory,
                input_transform=Normalize(  # TODO: Normalization causes issue with condition_on_observations
                    d=self._state_size + 1,
                    indices=list(range(self._state_size))),  # ONLY normalize state part.
                outcome_transform=None,
                use_scale_kernel=self.use_scale_kernel
            ).to(self.device)
        elif self._gp_mode == 'variational_gp':
            # Do not use RFF with variational GP.
            if train_x.shape[0] < self._num_inducing_points:
                return MixedSingleTaskVariationalGP(
                    train_X=train_x,
                    train_Y=train_y,
                    cat_dims=[self._state_size],
                    cont_kernel_factory=self._kernel_factory,
                    inducing_points=self._num_inducing_points,
                    input_transform=Normalize(d=self._state_size + 1,
                                              indices=list(range(self._state_size))),
                    outcome_transform=None
                ).to(self.device)
            else:
                return self._current_gp
        elif self._gp_mode == 'deep_gp':
            if first_time:
                # print(list(range(self._state_size - 2)))

                dpg = DeepGPModel(
                    train_x_shape=train_x.shape,
                    hidden_layers_config=Config.DGP_HIDDEN_LAYERS_CONFIG,
                    # hidden_layers_config=[
                    #    {"output_dims": 1, "mean_type": "linear"},
                    #    {"output_dims": 2, "mean_type": "linear"},
                    #    # {"output_dims": 1, "mean_type": "linear"},
                    #    {"output_dims": None, "mean_type": "constant"}
                    # ],
                    cat_dims=[self._state_size],
                    num_inducing_points=self._num_inducing_points,
                    #input_transform=Normalize(d=self._state_size + 1,
                    #                          indices=list(range(self._state_size - 2)))
                ).to(self.device)
                self._optimizer = torch.optim.Adam(dpg.parameters(), lr=Config.GP_FIT_LEARNING_RATE)
                return dpg
            else:
                return self._current_gp.to(self.device)

    def fit(self, new_train_x: torch.tensor, new_train_y: torch.tensor, hyperparameter_fitting: bool = True) -> None:
        """
        Condition the GP on incoming data and fit its hyperparameters.
        Sets the _current_gp private field.

        :param new_train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param new_train_y: A tensor with expected shape (dataset_size, 1).
        :param hyperparameter_fitting: Whether to fit the kernel hyperparameters or not.
        """
        with botorch.settings.debug(True):
            if self._exploring_starts > 0:
                return

            self._dummy_counter += 1

            # print(new_train_x)

            self.extend_dataset(new_train_x, new_train_y)
            train_x, train_y = self.dataset()
            gp = self._construct_gp(train_x, train_y)

            # print("DEVICE DATA:", train_x.device.type)
            # print("DEVICE GP", gp.device.type)

            print("Dataset size ->", train_x.shape[0])
            if hyperparameter_fitting:
                start_time = time.time()
                checkpoint_path = None if self._gp_mode == 'deep_gp' else 'gp_model_checkpoint_' + self._gp_mode + '.pth'
                self.latest_loss = self.fit_gp(gp, train_x, train_y, self._gp_mode,
                                               batch_size=Config.GP_FIT_BATCH_SIZE,
                                               num_epochs=Config.GP_FIT_NUM_EPOCHS,
                                               num_batches=Config.GP_NUM_BATCHES,
                                               learning_rate=Config.GP_FIT_LEARNING_RATE,
                                               random_batching=Config.GP_FIT_RANDOM_BATCHING,
                                               logging=True,
                                               checkpoint_path=checkpoint_path,
                                               optimizer=self._optimizer)
                logger.debug(f"Time taken -> {time.time() - start_time} seconds")

            self._current_gp = gp

    def get_current_gp(self):
        """
        :return: The current Gaussian process model.
        """
        return self._current_gp

    def _linear_independence_test(self,
                                  test_point: torch.tensor,
                                  data_points: torch.tensor,
                                  kernel_matrix: torch.tensor,
                                  kernel_fun) -> bool:
        """
        Computes whether the linear independence test is passed.
        The linear independence test asks, how well is test_point approximated by the elements
        of data_points?
        NOTE: DOES NOT APPEAR TO WORK, THRESHOLD NEVER GETS REACHED AND THE VALUES
        THAT ARE COMPUTED ARE OFTEN SMALL/CLOSE TO ZERO.
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
        """
        Add data points to the dataset using a sparsification scheme.

        :param new_train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param new_train_y: A tensor with expected shape (dataset_size, 1).
        """
        covar_fun = self._current_gp.covar_module
        train_x = torch.cat(list(self._data_x))
        covar_matrix = covar_fun(train_x, train_x)

        for x, y in zip(new_train_x, new_train_y):
            if self._linear_independence_test(x.unsqueeze(0), train_x, covar_matrix, covar_fun):
                self._data_x.append(x.unsqueeze(0))
                self._data_y.append(y.unsqueeze(0))

    def extend_dataset(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        """
        Add data points to the dataset.

        :param new_train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param new_train_y: A tensor with expected shape (dataset_size, 1).
        """
        # new_train_x = self._current_gp.transform_inputs(new_train_x)

        # print("->", new_train_x)

        # dequeue evaluates to False if empty.
        if self._sparsification_treshold is not None and self._data_x:
            self._sparse_add(new_train_x, new_train_y)
            return

        # Maybe there is a better way to do this
        # We want to tensors to be non-batched in the deque.
        for x, y in zip(new_train_x, new_train_y):
            self._data_x.append(x.unsqueeze(0))
            self._data_y.append(y.unsqueeze(0))

    def dataset(self) -> tuple[torch.tensor, torch.tensor]:
        """
        Construct batched tensors of shape (dataset_size, state_space_dim) and (dataset_size, 1),
        from the dequeue for inputs and targets.
        :return: a batched tensors of shape (dataset_size, state_space_dim) and (dataset_size, 1),
        """
        train_x = torch.cat(list(self._data_x))
        train_y = torch.cat(list(self._data_y))

        return train_x, train_y

    def choose_next_action(self, state: torch.tensor) -> int:
        """
        Choose the next action based on the behavioral policy with the current GP model.

        Choosing the next point works a bit differently compared to traditional BayesOpt.
        We must fix the state part of the input :math: `z \in S times A`
        and search for the most suitable action part.
        Target is typically TD(0) target.
        :param state: a state tensor (1, state_space_dim).
        :return: an action encoded as an int.
        """
        if self._exploring_starts > 0:
            self._exploring_starts -= 1
            return self._action_space.sample()

        self._gp_action_selector.update()

        action_tensor = self._gp_action_selector.action(self._current_gp, state)

        if self._dummy_counter == 50:
            self._visualize_data(state)

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
            q_val = self._current_gp.posterior(next_state_action_pairs,
                                               observation_noise=self._posterior_obs_noise).mean

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

                mean_qs = self._current_gp.posterior(state_action_pairs,
                                                     observation_noise=self._posterior_obs_noise).mean
                # batch_size amount of q_values.
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

    def _visualize_data(self, state):
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

    def exploring_starts(self) -> int:
        return self._exploring_starts
