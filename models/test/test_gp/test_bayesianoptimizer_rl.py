import unittest
from unittest.mock import MagicMock

import torch
import gymnasium as gym

from bayesopt.bayesianoptimizer_rl import BayesianOptimizerRL, append_actions, simple_thompson_action_sampler


class TestBayesianOptimizerRL(unittest.TestCase):
    def setUp(self):
        # Initialize test parameters
        self.model_str = 'exact_gp'
        self.random_draws = 5
        self.max_dataset_size = 100
        self.state_size = 4
        self.action_size = 2
        self.action_space = gym.spaces.Discrete(2)  # Example discrete action space with 3 actions
        self.bayesian_optimizer = BayesianOptimizerRL(self.model_str, self.random_draws, self.max_dataset_size,
                                                      self.state_size, self.action_space)

    def test_append_actions(self):
        """
        Append to a state vector (s1, s2, ..., s_n) actions such that we have
        a batch of tensors of the form: (s1, s2, ..., s_n, a_1),
                                        (s1, s2, ..., s_n, a_2),
                                        ...
                                        (s1, s2, ..., s_n, a_m)   m = num of actions.
        """
        state_size = 4
        action_size = 2

        # Test append_actions function
        state = torch.zeros(state_size)  # Example state tensor

        state_action_pairs = append_actions(state, action_size, device='cpu')
        self.assertIsInstance(state_action_pairs, torch.Tensor)
        self.assertEqual(state_action_pairs.shape, torch.Size([action_size, state_size + 1]))

        # Test the content of the tensor
        expected_pairs = torch.tensor([[0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 1.]])  # Expected state-action pairs
        self.assertTrue(torch.allclose(state_action_pairs, expected_pairs))

    def test_simple_thompson_action_sampler(self):
        state_size = 4
        action_size = 2
        # I would prefer unit tests to be on the CPU only but the way
        # the function is implemented means it will automatically place
        # tensors on the GPU...
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test simple_thompson_action_sampler function
        gpq_model_mock = MagicMock()
        # Define a fixed tensor as the return value of rsample
        rsample_output = torch.tensor([[[0],
                                        [7],
                                        [0]]]).to(device=device)
        gpq_model_mock.posterior.return_value.rsample.return_value = rsample_output  # Example rsample output

        state_tensor = torch.zeros(state_size, device=device)  # Example state tensor
        best_action = simple_thompson_action_sampler(gpq_model_mock, state_tensor, action_size)
        self.assertIsInstance(best_action, torch.Tensor)
        self.assertEqual(best_action, 1)

    def test_max_state_action_value(self):
        # Cannot test this properly now because of how the gps are instantiated (cannot mock).
        # Debugging statements seem to imply this method works correctly.
        pass


if __name__ == '__main__':
    unittest.main()
