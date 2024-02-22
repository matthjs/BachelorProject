from typing import Tuple
import numpy as np

"""
Online algorithms for keeping track of sample means and sample variances.
At each time step we want to keep track of the sample mean and sample variance seen so far.

For more information, see: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

.. note::
   This module implements online algorithms for calculating mean, variance, and sample variance.
"""


class Welford:
    @staticmethod
    def update_aggr(existing_aggr: Tuple[int, float, float], new_val: float) -> Tuple[int, float, float]:
        """
        Update aggregate values with a new value.

        :param existing_aggr: Tuple containing the current aggregate values (count, mean, M2).
        :param new_val: The new value to be incorporated into the aggregate.
        :return: Updated aggregate values after incorporating the new value.
        """
        (count, mean, M2) = existing_aggr
        count += 1
        delta = new_val - mean
        mean += delta / count
        delta2 = new_val - mean
        M2 += delta * delta2
        return count, mean, M2

    @staticmethod
    def finalize_aggr(existing_aggr: Tuple[int, float, float]) -> Tuple[float, float, float]:
        """
        Retrieve the mean, variance, and sample variance from an aggregate.

        :param existing_aggr: Tuple containing the aggregate values (count, mean, M2).
        :return: Mean, variance, and sample variance calculated from the aggregate.
        """
        count, mean, m2 = existing_aggr
        if count < 2:
            return mean, 0, 0
        else:
            variance = m2 / count
            sample_variance = m2 / (count - 1)
            return mean, variance, sample_variance
