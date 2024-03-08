import gpytorch.mlls.exact_marginal_log_likelihood
import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal

from agent.abstractdpagent import AbstractDPAgent
import gymnasium as gym

from models.gp import GaussianProcessRegressor


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




class GaussianProcessDPAgent(AbstractDPAgent):
    """
    Dynamic Programming Agent.
    TODO: Refactor the class so that the training steps are decoupled from the agent.
    """

    def __init__(self,
                 env: gym.Env,
                 discount_factor=0.9,
                 dynamics_fit_iter=100,
                 learning_rate=0.01,
                 batch_num_support_points=50):
        # Awkward: passing empty dictionary to base class -> class hierarchy is a bit messy.
        super().__init__({}, env.observation_space, env.action_space)
        self.policy_calculated = False
        self.env = env
        self.dynamics_fit_iter = dynamics_fit_iter
        self.marginal_likelihood = gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood
        self.dynamics_gps = None
        self.value_gp = None
        self.optimizer_type = torch.optim.Adam
        self.learning_rate = learning_rate
        self.batch_num_support_points = batch_num_support_points

    def _simulate(self, time_interval=5) -> tuple:
        """
        Obtain observations of the system dynamics.
        :param time_interval:
        :return: a tuple train_X (s,a) and train_y (s'). For the mountain car problem
        s is 2-dimensional real vector.
        """
        train_x = []
        train_y = []

        obs, info = self.env.reset()
        for _ in range(time_interval):
            old_obs = obs
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            vec_x = np.append(old_obs, action)
            train_x.append(torch.tensor(vec_x))
            train_y.append(torch.tensor(obs))

        return torch.stack(train_x), torch.stack(train_y)

    def _fit_gaussian_processes(self, train_x, train_y):
        """
        Fit a gaussian process for every output dimension.
        Fitting=fit the kernel hyperparameters.
        """
        models = []

        for idx in range(train_y.size(-1)):
            # Model the system dynamics by Gaussian processes for each state coordinate
            # and combine them to obtain a model of the transition function (as a multivariate normal).
            model = GaussianProcessRegressor(train_x, train_y[..., idx])
            model.train()  # Enable 'training' mode for MLE of hyperparameters of kernel.
            model.likelihood.train()
            mll = self.marginal_likelihood(model.likelihood, model)
            optimizer = self.optimizer_type(model.parameters(), lr=self.learning_rate)

            for _ in range(self.dynamics_fit_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y[..., idx])  # The loss is the negative likelihood.
                loss.backward()
                optimizer.step()

            models.append(model)

        return models

    def _fit_value_function(self, state_support_points):
        value_vector = []
        for state in state_support_points:
            pass
            # How to get the reward function. Also in the paper the reward function appears to be a
            # a function of just the state. For the mountain car problem the reward function is simple though.


    def iterate(self):
        self.policy_calculated = True
        train_x, train_y = self._simulate()
        self.dynamics_gps: list[GaussianProcessRegressor] = self._fit_gaussian_processes(train_x, train_y)

        # The idea: select a finite number of support points and let the
        # GP generalize to the entire space.
        state_support_points = torch.tensor(
            np.linspace(self.env.observation_space.low,
                        self.env.observation_space.high,
                        num=self.batch_num_support_points))
        #print(state_support_points)

        # Value iteration
        while True:
            for idx, state in enumerate(state_support_points):
                # action =

    def policy(self, state):
        if self.policy_calculated is False:
            print("Optimal Policy Not calculated yet, running policy iteration")
            self.iterate()
