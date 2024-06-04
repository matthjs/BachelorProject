import unittest
from unittest.mock import Mock

import torch
from gp.mixedvariationalgp import max_state_action_value


class TestMaxStateActionValue(unittest.TestCase):
    def setUp(self):
        # Create a mock QGP model
        # Initialize a dummy QGP model
        # self.qgp_model = ExactGaussianProcessRegressor(likelihood=gpytorch.likelihoods.GaussianLikelihood())
        # Create a mock QGP model
        self.qgp_model = Mock()
        # Define the side effect to return different values on each call
        self.qgp_model.predict.side_effect = [([torch.tensor([[0.5, 0, 1, 2, 0.1]]), None, None, None]),
                                              ([torch.tensor([[0.6, 2, 0, 0.1, 0.3]]), None, None, None]),
                                              ([torch.tensor([[0.1, 0.2, 0.3, 0.4, 1]]), None, None, None])
                                              ]
        # Define some dummy data
        self.action_space_size = 3
        self.state_batch = torch.randn(5, 2)  # Example: batch size 5, state vector size 2

    def test_max_state_action_value(self):
        # Call the function
        max_q_values, max_actions = max_state_action_value(self.qgp_model, self.action_space_size, self.state_batch, "cpu")
        # Assert that the shape of the result matches the expected shape
        self.assertEqual(max_q_values.shape, (5, 1))  # Expected shape: (batch_size, 1)
        print(max_actions[0].squeeze().numpy())


if __name__ == '__main__':
    unittest.main()
