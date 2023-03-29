'''
  自己写的对环境的类包装
'''
import gym
import numpy as np
SPARSE_MOUNTAIN_CAR = "SparseMountainCar-v0"
PENDULUM = "Pendulum-v1"
FROZENLAKE = "FrozenLake-v1"


class GymEnv(object):

    def __init__(self, env_name, max_episode_len, action_repeat = 1, seed = None):
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.env = self._get_env_object(env_name)
        self.t = 0

    def reset(self):
        state = self.env.reset()
        self.t = 0
        return state

    def step(self, action):
        global state, done, info
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, info = self.env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len  # 运算符优先级，计算顺序为 done  = (done or (self.t == self.max_episode_len))
            if done:
                self.done = True
                break
        return state, reward, done, info

    def sample_action(self):
        return self.env.sample_action()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.env   #这里必须是env.env才可以生成视频


    def _get_env_object(self, env_name):

        if env_name == SPARSE_MOUNTAIN_CAR:

            from pmbrl.envs.envs.mountain_car import SparseMountainCarEnv

            return SparseMountainCarEnv()

        if env_name == PENDULUM:

            from pmbrl.envs.envs.Pendulum import Pendulum

            return Pendulum()

        if env_name == FROZENLAKE:

            from pmbrl.envs.envs.frozen_lake import FrozenLake

            return FrozenLake()
