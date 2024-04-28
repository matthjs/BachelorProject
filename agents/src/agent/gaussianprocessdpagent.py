import gpytorch.mlls.exact_marginal_log_likelihood
import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from loguru import logger

from agent.abstractdpagent import AbstractDPAgent
import gymnasium as gym
import torch

from models.gp import ExactGaussianProcessRegressor
from trainers.gptrainer import GaussianProcessTrainer
from util.fetchdevice import fetch_device

from wrappers.initialstatewrapper import InitialStateWrapper


def mountain_car_reward_fun(state) -> torch.tensor:
    """
    Kinda have to do this since the gymnasium reward function is not a function of just the state but
    the state and action r(s,a). TODO: Find a more flexible approach.
    :return: scalar reward value.
    """
    if state[0] < 0.5:
        return torch.tensor(-1.0, device=fetch_device())
    else:
        # Goal reached if x-pos is >= 0.5.
        return torch.tensor(0.0, device=fetch_device())


class GaussianProcessDPAgent(AbstractDPAgent):
    """
    Dynamic Programming Agent.
    """

    def __init__(self,
                 env: gym.Env,
                 discount_factor=0.9,
                 dynamics_fit_iter=50,
                 value_fit_iter=50,
                 learning_rate=0.01,
                 batch_num_support_points=50,
                 simulation_time_interval=5,
                 matern_kernel_lengthscale=2,
                 matern_kernel_mu=1.5,
                 noise_level=1):
        # Awkward: passing empty dictionary to base class -> class hierarchy is a bit messy.
        super().__init__({}, env.observation_space, env.action_space)
        self.env = InitialStateWrapper(env)  # Needed to set the state of the environment.
        self.dynamics_fit_iter = dynamics_fit_iter
        self.value_fit_iter = value_fit_iter

        self.dynamics_gps = None
        self.transition_model = None

        self.value_gp: ExactGaussianProcessRegressor = None
        self.value_vector = None
        self.state_support_points = None

        self.kernel = ScaleKernel(MaternKernel(nu=matern_kernel_mu))
        # self.likelihood = FixedNoiseGaussianLikelihood(noise=torch.ones())

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_num_support_points = batch_num_support_points
        self.simulation_time_interval = simulation_time_interval
        self.reward_fun = mountain_car_reward_fun

    def _simulate(self) -> tuple:
        """
        Obtain observations of the system dynamics.
        :param time_interval:
        :return: a tuple train_X (s,a) and train_y (s'). For the mountain car problem
        s is 2-dimensional real vector.
        """
        train_x = []
        train_y = []

        obs, info = self.env.reset()
        for _ in range(self.simulation_time_interval):
            old_obs = obs
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            vec_x = np.append(old_obs, action)
            train_x.append(torch.tensor(vec_x, device=fetch_device()))
            train_y.append(torch.tensor(obs, device=fetch_device()))

        return torch.stack(train_x).to(device=fetch_device()), torch.stack(train_y).to(device=fetch_device())

    def _fit_gaussian_processes(self, train_x, train_y):
        """
        Fit a gaussian process for every output dimension.
        Fitting=fit the kernel hyperparameters.
        """
        models = []

        for idx in range(train_y.size(-1)):
            logger.debug("Fitting Gaussian Process kernel hyperparams for dynamics")
            # Model the system dynamics by Gaussian processes for each state coordinate
            # and combine them to obtain a model of the transition function (as a multivariate normal).
            model = ExactGaussianProcessRegressor(train_x, train_y[..., idx]).to(device=fetch_device())
            trainer = GaussianProcessTrainer(model, learning_rate=self.learning_rate)
            trainer.train(train_x, train_y[..., idx], num_epochs=self.dynamics_fit_iter)

            models.append(model)

        return models

    def _construct_dynamics_model(self, train_x: torch.tensor, train_y) -> None:
        """
        Model the transition function as a multivariate normal where the mean vector and covariance
        matrix come from the Bayesian predictive posterior from two Gaussian Processes.
        One GP for each output dimension.
        :param train_x:
        :param train_y:
        """
        self.dynamics_gps: list[ExactGaussianProcessRegressor] = self._fit_gaussian_processes(train_x, train_y)

        means = []
        covars = []

        for model in self.dynamics_gps:
            model.double()  # Why the hell do I need to do this?
            _, _, _, f_pred = model.predict(train_x)
            means.append(f_pred.mean)
            covars.append(f_pred.covariance_matrix)

        combined_mean = torch.cat(means, dim=-1)
        combined_covar = torch.block_diag(*covars)
        # Is this what the paper also does? It is a bit vague what they mean with `combining`.

        # print(combined_mean)
        # print(combined_covar)
        self.transition_model = torch.distributions.MultivariateNormal(combined_mean, combined_covar)

    def _fit_value_function(self):
        """
        Given a set of support points s_i we initialize the value vector as V_i <- Reward_fun(s_i).
        Afterward a Gaussian process models is used to model V(s). HyperParameters are also fitted
        using the marginal log likelihood.
        :param self.state_support_points:
        """
        value_vector = []
        for state in self.state_support_points:
            reward = self.reward_fun(state)
            value_vector.append(reward)

        self.value_vector = torch.tensor(value_vector)
        value_vector = self.value_vector.to(device=fetch_device())
        support_points = self.state_support_points.to(device=fetch_device())

        self.value_gp = ExactGaussianProcessRegressor(train_x=support_points,
                                                      train_y=value_vector,
                                                      covar_function=self.kernel).to(fetch_device())
        logger.debug("Fitting Gaussian Process kernel hyperparameters for value function")
        # Fit Gaussian process hyperparameters for representing V(s).
        trainer = GaussianProcessTrainer(self.value_gp, learning_rate=self.learning_rate)
        trainer.train(train_x=support_points, train_y=value_vector,
                      num_epochs=self.value_fit_iter)

    def _greedy_action_selection(self, state):
        """
        Perform policy improvement step.
        https://github.com/openai/gym/issues/402.
        This is done by making the policy greedy w.r.t. to the
        current value function estimate.
        """
        # argmax loop.
        argmax_action = None
        argmax_value = -999

        assert (isinstance(self.env.action_space, gym.spaces.Discrete))

        for action in range(self.env.action_space.n):
            obs = self.env.reset(initial_state=state)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs_tensor = torch.as_tensor(next_obs, device=fetch_device())
            next_obs_tensor = next_obs_tensor.reshape(1, next_obs_tensor.shape[0])
            # print(next_obs_tensor)
            # print(next_obs_tensor.shape)
            self.value_gp.double()  # Why?!
            value, lower, upper, f_pred = self.value_gp.predict(next_obs_tensor)

            action_value = reward + self.discount_factor * value

            if action_value > argmax_value:
                argmax_action = action
                argmax_value = action_value

        return argmax_action

    def _sample_support_points(self):
        logger.debug("Calculating state support points.")
        """
        I say "sample" but there is no sampling. it is a deterministic state
        space grid.
        """
        self.state_support_points = torch.tensor(
            np.linspace(self.env.observation_space.low,
                        self.env.observation_space.high,
                        num=self.batch_num_support_points))

    def iterate(self):
        self.policy_calculated = True
        train_x, train_y = self._simulate()
        self._construct_dynamics_model(train_x, train_y)

        # The idea: select a finite number of support points and let the
        # GP generalize to the entire space.
        self._sample_support_points()

        self._fit_value_function()

        reward_vec = np.zeros(self.batch_num_support_points)
        value_gp_trainer = GaussianProcessTrainer(self.value_gp, learning_rate=self.learning_rate)

        # Value iteration
        while True:
            old_value_vector = self.value_vector.clone().to(device=fetch_device(), dtype=torch.float64)

            for idx, state in enumerate(self.state_support_points):
                action = self._greedy_action_selection(state)
                # Very strange... in the paper implementation they do not use the dynamics GP
                self.env.reset(initial_state=state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # As a consequence the reward is simply given by the environment and need not be computed
                reward_vec[idx] = reward

                state_tensor = torch.as_tensor(state, device=fetch_device())
                state_tensor = state_tensor.reshape(1, state_tensor.shape[0])
                state_value, _, _, _ = self.value_gp.predict(state_tensor)

                self.value_vector[idx] = reward + self.discount_factor * state_value

            # Update the GP hyperparameters for representing V(s) to fit the new value vector V.
            self.value_gp.train()
            state_support_points = self.state_support_points.to(device=fetch_device(), dtype=torch.float64)
            value_vector = self.value_vector.to(device=fetch_device(), dtype=torch.float64)
            logger.debug(f"Value vector -> {self.value_vector}")

            self.value_gp.set_train_data(inputs=state_support_points,
                                         targets=value_vector)
            value_gp_trainer.train(train_x=state_support_points,
                                   train_y=value_vector,
                                   num_epochs=self.value_fit_iter,
                                   logging=False)

            # Check for convergence
            if torch.allclose(old_value_vector, value_vector, atol=1e-3):
                break

    def policy(self, state):
        if self.policy_calculated is False:
            logger.info("Optimal Policy Not calculated yet, running policy iteration")
            self.iterate()

        return self._greedy_action_selection(state)

