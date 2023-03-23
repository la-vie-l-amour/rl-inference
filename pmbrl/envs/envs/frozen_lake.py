import gymnasium as gym

class FrozenLake:
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self):
        gym.make("FrozenLake-v1")
