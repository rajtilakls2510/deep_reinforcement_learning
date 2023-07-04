import gymnasium as gym
from deep_rl.agent import GymEnvironment
import numpy as np


class AntEnvironment(GymEnvironment):
    def __init__(self, env: gym.Env):
        super(AntEnvironment, self).__init__(env)
        # self.theta = 0.15
        # self.mean = np.zeros(1)
        self.std_dev = float(0.2) * np.ones(1)
        # self.dt = 1e-2
        # self.x = np.zeros_like(self.mean)

    def get_random_action(self):
        return np.float32(np.random.normal(size=(3,), loc=0.0, scale=0.1))
        # self.x = (
        #         self.x
        #         + self.theta * (self.mean - self.x) * self.dt
        #         + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        # )
        # return self.x

    def take_action(self, action):
        action = np.clip(action, -1, 1)
        super(AntEnvironment, self).take_action(action)

