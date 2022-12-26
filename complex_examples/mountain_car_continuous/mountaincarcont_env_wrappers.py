import gym
from deep_rl.agent import GymEnvironment
import numpy as np


class MountainCarContinuousEnvironment(GymEnvironment):
    def __init__(self, env: gym.Env):
        super(MountainCarContinuousEnvironment, self).__init__(env)
        self.theta = 0.15
        self.mean = np.zeros(1)
        self.std_dev = float(0.2) * np.ones(1)
        self.dt = 1e-2
        self.x = np.zeros_like(self.mean)

    def get_random_action(self):
        # return random.normal(shape=(1,), mean=0.0, stddev=0.2, dtype=float32)
        self.x = (
                self.x
                + self.theta * (self.mean - self.x) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        return self.x

    def take_action(self, action):
        action = np.clip(action, -1, 1)
        super(MountainCarContinuousEnvironment, self).take_action(action)

    def calculate_reward(self):
        vel = abs(self.state[1]) * 1_000
        if vel < 0.1:
            vel = 0.1

        reward = (self.state[0] - 0.5) / vel
        if self.state[0] >= 0.5:
            reward = 100.0
        return reward
