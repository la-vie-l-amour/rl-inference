import math

import gymnasium as gym
import numpy as np

# 连续动作大都是关于0对称的,所以这里把动作包装成关于0对称
class FrozenLake:
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self,render_mode = "rgb_array"):
        self.env = gym.make("FrozenLake-v1",render_mode = render_mode)
        self.render_mode = render_mode


    def reset(self):
        state,_ = self.env.reset()
        state = np.array([state])
        return state

    def step(self, action):
        action = np.clip(action,-1,2)
        action = math.ceil(action[0]) + 1
        state, reward, term, trun, info= self.env.step(action)
        done = np.bitwise_or(term, trun)
        state = np.array([state])
        return state, reward, done, info

    def close(self):
        self.env.close()
    def render(self):
        self.render()

    @property
    def action_space(self):
        # return np.zeros(1)
        return gym.spaces.Box(low = -2, high = 2,shape=(1,), dtype= np.int32)

    @property
    def observation_space(self):
        # return np.zeros(1)
        return gym.spaces.Box(low = 0, high = 15, shape = (1,),dtype = np.int32)

    def sample_action(self):
        return np.array([self.env.action_space.sample()])





