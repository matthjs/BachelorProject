from metricstracker.metricstracker4 import MetricsTracker4


class MetricsTrackerRegistry:
    def __init__(self):
        self._trackers = {}

    # _instance = None

    """
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._trackers = {}

        return cls._instance
    """

    def register_tracker(self, tracker_id: str):
        self._trackers[tracker_id] = MetricsTracker4()

    def get_tracker(self, tracker_id: str):
        if tracker_id not in self._trackers:
            raise ValueError("tracker_id not present.")

        return self._trackers[tracker_id]
