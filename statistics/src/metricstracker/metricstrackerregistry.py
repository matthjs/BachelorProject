from metricstracker.metricstracker import MetricsTracker


class MetricsTrackerRegistry:
    """
    Registry for managing multiple MetricsTracker instances.
    """

    def __init__(self):
        """
        Initialize the MetricsTrackerRegistry instance.
        """
        self._trackers = {}

    def register_tracker(self, tracker_id: str) -> None:
        """
        Register a new tracker with the given ID.

        :param tracker_id: The ID of the tracker to register.
        """
        self._trackers[tracker_id] = MetricsTracker()

    def get_tracker(self, tracker_id: str) -> MetricsTracker:
        """
        Get the tracker with the specified ID.

        :param tracker_id: The ID of the tracker to retrieve.
        :return: The MetricsTracker instance associated with the given ID.
        :raises ValueError: If the tracker_id is not present in the registry.
        """
        if tracker_id not in self._trackers:
            raise ValueError("tracker_id not present.")

        return self._trackers[tracker_id]