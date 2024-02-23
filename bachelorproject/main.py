from loops.envinteraction import env_interaction

if __name__ == "__main__":
    env_interaction("none", "MountainCar-v0", "none", 100, "human")
    """
    MetricsTracker().record_loss("cat", 2)
    MetricsTracker().record_loss("cat", 4)
    MetricsTracker().record_reward("cat", 24)
    MetricsTracker().record_reward("cat", 2)
    print(MetricsTracker().loss_history["cat"])
    print(MetricsTracker().reward_history["cat"])
    """
