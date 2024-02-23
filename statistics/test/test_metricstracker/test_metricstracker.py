import random
import threading
import unittest

from metricstracker.metricstracker import MetricsTracker


class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        # Create an instance of MetricsTracker for testing
        self.metrics_tracker = MetricsTracker()


if __name__ == "__main__":
    unittest.main()
