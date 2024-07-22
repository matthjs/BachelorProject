from collections import deque
import time
import botorch.settings
import torch
from botorch.models.gpytorch import GPyTorchModel
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from bachelorproject.configobject import Config
from bayesopt.abstractbayesianoptimizer_rl import AbstractBayesianOptimizerRL
from bayesopt.acquisition import ThompsonSampling, \
    UpperConfidenceBound, GPEpsilonGreedy
from gp.custommixedgp import MixedSingleTaskGP
from gp.deepgp import DeepGPModel
from gp.fitgp import GPFitter
from gp.gpviz import plot_gp_point_distribution
from kernels.kernelfactory import create_kernel
from util.fetchdevice import fetch_device
import gymnasium as gym


class BayesianOptimizerRL(AbstractBayesianOptimizerRL):
    """
    Bayesian optimizer for model-free RL with discrete action spaces.
    WARNING: This class is a bit big, due to not strictly following the
    one class one responsibility principle.
    """

    def __init__(self,
                 model_str: str,
                 exploring_starts: int,
                 max_dataset_size: int,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 kernel_type: str,
                 kernel_args,
                 strategy='thompson_sampling',
                 posterior_observation_noise: bool = False,
                 num_inducing_points: int = 64
                 ):
        """
        Constructor.
        NOTE: Constructor too long.

        :param model_str: What GP model to use ('exact_gp', 'deep_gp').
        :param exploring_starts: Number of choose_next_action() calls before the GP starts getting updated.
        :param max_dataset_size: Max dataset size.
        :param state_space: The state space of the environment.
        :param action_space: The action space of the environment (has to be discrete).
        :param kernel_type: Type of kernel to be used for states ('matern', 'rff', 'rbf').
               NOTE/TODO: deep_gp currently has a hardcoded RBF kernel for each unit -> so this argument does not
               do anything in that case.
        :param kernel_args: Only applicable for RFF for specifying number of samples (is a data object).
        :param strategy: Acquisition function/behavioral policy to use.
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

        self._viz_counter = 0
        self._visualize = False

        self._kernel_factory, self.use_scale_kernel = create_kernel(kernel_type, kernel_args)

        if model_str not in ['exact_gp', 'deep_gp']:
            raise ValueError(f'Unknown gp model type: {model_str}')

        self.fit_gp = GPFitter()
        self.latest_loss = 0

        # Initialize GP settings. The GP is filled with a few dummy values.
        self._gp_mode = model_str
        self._current_gp = None
        self._lr_scheduler = None
        self._current_gp = self._construct_gp(torch.zeros(10, state_size + 1, dtype=torch.double),
                                              torch.zeros(10, 1, dtype=torch.double), first_time=True)

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
        For deep_gp, the train_x.shape is used to get the right architecture for the first layer.
        Side effect for deep_gp: Instantiates an Adam optimizer that is used in GPFitter.

        :param train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param train_y: A tensor with expected shape (dataset_size, 1).
        :return: A GP conditioned on (train_x, train_y) (Hyperparameters NOT fitted).
                For deep_gp, the DGP is only instantiated once and after which the same
                reference is returned every time.
        """
        if self._gp_mode not in ["deep_gp"]:
            del self._current_gp

        # NOTE: This is too memory intensive for a lot of RL environments.
        if self._gp_mode == 'exact_gp':
            return MixedSingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                cat_dims=[self._state_size],
                cont_kernel_factory=self._kernel_factory,
                use_scale_kernel=self.use_scale_kernel
            ).to(self.device)
        elif self._gp_mode == 'deep_gp':
            if first_time:
                dpg = DeepGPModel(
                    train_x_shape=train_x.shape,
                    hidden_layers_config=Config.DGP_HIDDEN_LAYERS_CONFIG,
                    num_inducing_points=self._num_inducing_points,
                ).to(self.device)

                # Hardcoded gamma, since for the experiment I do not actually use LR scheduling.
                self._lr_scheduler = ExponentialLR(
                    torch.optim.Adam(dpg.parameters(), lr=Config.GP_FIT_LEARNING_RATE), gamma=1)
                return dpg
            else:
                return self._current_gp.to(self.device)

    def fit(self, new_train_x: torch.tensor, new_train_y: torch.tensor, hyperparameter_fitting: bool = True) -> None:
        """
        Condition the GP on incoming data and fit its hyperparameters.
        More accurately, for DGP, the incoming data is used, to adjust hyperparameters, including
        the inducing points, which is what the DGP actually conditions on.
        Sets the _current_gp private field.

        :param new_train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param new_train_y: A tensor with expected shape (dataset_size, 1).
        :param hyperparameter_fitting: Whether to fit the kernel hyperparameters or not.
        """
        with botorch.settings.debug(True):
            self.extend_dataset(new_train_x, new_train_y)
            if self._exploring_starts > 0:
                return
            self._viz_counter += 1
            train_x, train_y = self.dataset()
            gp = self._construct_gp(train_x, train_y)

            print("Dataset size ->", train_x.shape[0])
            if hyperparameter_fitting:
                start_time = time.time()
                checkpoint_path = None if self._gp_mode == 'deep_gp' else 'gp_model_checkpoint_' + self._gp_mode + '.pth'
                optimizer = self._lr_scheduler.optimizer if self._gp_mode == 'deep_gp' else None
                self.latest_loss = self.fit_gp(gp, train_x, train_y, self._gp_mode,
                                               batch_size=Config.GP_FIT_BATCH_SIZE,
                                               num_epochs=Config.GP_FIT_NUM_EPOCHS,
                                               num_batches=Config.GP_NUM_BATCHES,
                                               learning_rate=Config.GP_FIT_LEARNING_RATE,
                                               random_batching=Config.GP_FIT_RANDOM_BATCHING,
                                               logging=True,
                                               checkpoint_path=checkpoint_path,
                                               optimizer=optimizer)
                logger.debug(f"Time taken -> {time.time() - start_time} seconds")
                self._lr_scheduler.step()  # Learning rate schedule update.

            self._current_gp = gp

    def get_current_gp(self):
        """
        :return: The current Gaussian process model.
        """
        return self._current_gp

    def extend_dataset(self, new_train_x: torch.tensor, new_train_y: torch.tensor) -> None:
        """
        Add data points to the dataset.

        :param new_train_x: A tensor with expected shape (dataset_size, state_space_dim).
        :param new_train_y: A tensor with expected shape (dataset_size, 1).
        """
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
        Side effect: Occasionally plots the predictive distribution for a given state.

        :param state: a state tensor (1, state_space_dim).
        :return: an action encoded as an int.
        """
        if self._exploring_starts > 0:
            self._exploring_starts -= 1
            return self._action_space.sample()

        self._gp_action_selector.update()

        action_tensor = self._gp_action_selector.action(self._current_gp, state)

        if self._viz_counter == 300:  # TODO: Hardcoded: this should be configurable.
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

                mean_qs = self._current_gp.posterior(state_action_pairs,
                                                     observation_noise=self._posterior_obs_noise).mean
                # batch_size amount of q_values.
                q_values.append(mean_qs)

            # Some reshaping black magic to get the max q value along each batch dimension.
            q_tensor = torch.cat(q_values, dim=0).view(len(q_values), -1, 1)
            max_q_values, max_actions = torch.max(q_tensor, dim=0)

        return max_q_values, max_actions

    def _visualize_data(self, state: torch.tensor) -> None:
        """
        Visualizes the predictive distribution for a state by plotting a Gaussian for every action.
        Resets self._viz_counter.
        :param state: current state.
        """
        plot_gp_point_distribution(self._current_gp,
                                   state,
                                   self._action_size,
                                   title=f'Point Distribution for state ({state})')
        self._viz_counter = 0

    def exploring_starts(self) -> int:
        return self._exploring_starts
