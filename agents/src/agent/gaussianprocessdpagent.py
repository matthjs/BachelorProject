import gpytorch.mlls.exact_marginal_log_likelihood
import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal
from loguru import logger

from agent.abstractdpagent import AbstractDPAgent
import gymnasium as gym
import torch
from models.gp import GaussianProcessRegressor
from trainers.gptrainer import GaussianProcessTrainer
from util.fetchdevice import fetch_device
from gpytorch import distributions


# Combine predictive models to obtain a multivariate Gaussian distribution
def predict_transition_distribution(models, state_action):
    predictive_distributions = []
    for model in models:
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_distribution = model(state_action)
        predictive_distributions.append(predictive_distribution)
    return MultivariateNormal(torch.cat([pred.mean.unsqueeze(-1) for pred in predictive_distributions], dim=-1),
                              torch.cat([pred.covariance_matrix.unsqueeze(0) for pred in predictive_distributions]))


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
    TODO: Refactor the class so that the training steps are decoupled from the agent.
    """

    def __init__(self,
                 env: gym.Env,
                 discount_factor=0.9,
                 dynamics_fit_iter=5,
                 value_fit_iter=5,
                 learning_rate=0.01,
                 batch_num_support_points=50,
                 simulation_time_interval=5):
        # Awkward: passing empty dictionary to base class -> class hierarchy is a bit messy.
        super().__init__({}, env.observation_space, env.action_space)
        self.policy_calculated = False
        self.env = env
        self.dynamics_fit_iter = dynamics_fit_iter
        self.value_fit_iter = value_fit_iter

        self.dynamics_gps = None
        self.transition_model = None

        self.value_gp = None
        self.value_vector = None

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
            # Model the system dynamics by Gaussian processes for each state coordinate
            # and combine them to obtain a model of the transition function (as a multivariate normal).
            model = GaussianProcessRegressor(train_x, train_y[..., idx]).to(device=fetch_device())
            logger.debug("Fitting Gaussian Process kernel hyperparams")
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
        self.dynamics_gps: list[GaussianProcessRegressor] = self._fit_gaussian_processes(train_x, train_y)

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

    def _fit_value_function(self, state_support_points):
        """
        Given a set of support points s_i we initialize the value vector as V_i <- Reward_fun(s_i).
        Afterward a Gaussian process models is used to model V(s). HyperParameters are also fitted
        using the marginal log likelihood.
        :param state_support_points:
        """
        value_vector = []
        for state in state_support_points:
            reward = self.reward_fun(state)
            value_vector.append(reward)

        self.value_vector = torch.tensor(value_vector)
        self.value_vector = self.value_vector.to(device=fetch_device())
        support_points = state_support_points.to(device=fetch_device())

        self.value_gp = GaussianProcessRegressor(train_x=support_points,
                                                 train_y=self.value_vector).to(fetch_device())

        # Fit Gaussian process hyperparameters for representing V(s).
        trainer = GaussianProcessTrainer(self.value_gp, learning_rate=self.learning_rate)
        trainer.train(train_x=support_points, train_y=self.value_vector, num_epochs=self.value_fit_iter)

    def _bellman_update(self, state):
        pass

    def iterate(self):
        self.policy_calculated = True
        train_x, train_y = self._simulate()
        self._construct_dynamics_model(train_x, train_y)

        # The idea: select a finite number of support points and let the
        # GP generalize to the entire space.
        state_support_points = torch.tensor(
            np.linspace(self.env.observation_space.low,
                        self.env.observation_space.high,
                        num=self.batch_num_support_points))

        self._fit_value_function(state_support_points)

        # Value iteration
        while True:
            for idx, state in enumerate(state_support_points):
                break
                # Compute equation 11
                # Compute transition probability
                # Compute R_i
                # Compute i-th row of W as in eq 9

            # Compute closed form of value vector.

            # Check for convergence
            # if torch.allclose(new_value_vector, self.value_vector, atol=1e-3):
            #    break

            # self.value_vector = new_value_vector

    def policy(self, state):
        if self.policy_calculated is False:
            print("Optimal Policy Not calculated yet, running policy iteration")
            self.iterate()
