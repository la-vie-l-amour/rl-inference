'''
  自己写的对环境的类包装
'''
import gym
import numpy as np

class Env(object):

    def __init__(self,env_name, max_episode_len, action_repeat = 1, seed = None):
        self.env = gym.make(env_name,render_mode = render_mode)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat

        self.env_name = env_name

    def reset(self):
        state, _ = self.env.reset()
        return state

    def SparseMountainCar_step(self,action):
        state, reward, term, trun, info = self.env.step(action)
        done = np.bitwise_or(term, trun)
        reward = 0
        if done:
            reward = 1.0
        return state, reward, done, info

    def Pendulum_step(self,action):
        state, reward, term, trun, info = self.env.step(action)
        done = np.bitwise_or(term, trun)
        return state, reward, done, info

    def step(self, action):
        global reward_k, done, state, info
        reward = 0
        for _ in range(self.action_repeat):

            if self.env_name =="Pendulum":
                state, reward_k, done, info = self.Pendulum_step(action)
            elif self.env_name == "SparseMountainCar":
                state, reward_k, done, info = self.SparseMountainCar_step(action)

            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len  # 运算符优先级，计算顺序为 done  = (done or (self.t == self.max_episode_len))
            if done:
                self.done = True
                break
        return state, reward, done, info




    def sample_action(self):
        return self.env.action_space.sample()

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
        return self.env


