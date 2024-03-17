import gymnasium as gym
import numpy as np

from wrappers.initialstatewrapper import InitialStateWrapper


def gym_test():
    env = gym.make("MountainCar-v0", render_mode="human")
    init_state, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.state = init_state

        if terminated or truncated:
            observation, info = env.reset()

    env.close()








if __name__ == "__main__":
    # Example usage
    env = gym.make("MountainCar-v0", render_mode="human")
    env = InitialStateWrapper(env)

    # Resetting with initial state
    initial_state = np.array([-1, 0.01])  # Example initial state
    obs = env.reset(initial_state=initial_state)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(np.array(2))
        print(reward)
        # env.state = init_state
        # env.reset(initial_state=initial_state)

        if terminated or truncated:
            observation, info = env.reset(initial_state=initial_state)

    # Now, the environment is initialized with the provided initial state
