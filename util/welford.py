from typing import Tuple

"""
Online algorithms for keeping track of sample means and sample variances.
At each time step we want to keep track of the sample mean and sample variance seen so far.

For more information, see: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

.. note::
   This module implements online algorithms for calculating mean, variance, and sample variance.
"""


class Welford:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_aggr(self, new_val: float) -> None:
        """
        Update aggregate values with a new value.

        :param new_val: The new value to be incorporated into the aggregate.
        """
        self.count += 1
        delta = new_val - self.mean
        self.mean += delta / self.count
        delta2 = new_val - self.mean
        self.M2 += delta * delta2

    def get_curr_mean_variance(self) -> Tuple[float, float]:
        """
        Return the current sample mean and sample variance estimate.
        :return: mean and variance.
        """
        mean, _, var = self._finalize_aggr()
        return mean, var

    def _finalize_aggr(self) -> Tuple[float, float, float]:
        """
        Retrieve the mean, variance, and sample variance from the aggregate.

        :return: Mean, variance, and sample variance calculated from the aggregate.
        """
        if self.count < 2:
            return self.mean, 0, 0
        else:
            variance = self.M2 / self.count
            sample_variance = self.M2 / (self.count - 1)
            return self.mean, variance, sample_variance
