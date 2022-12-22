import gym
from deep_rl.agent import GymEnvironment
from tensorflow import random, int32


class MountainCarEnvironment(GymEnvironment):

    def get_random_action(self):
        return random.uniform(shape=(), maxval=3, dtype=int32).numpy()


class MountainCarShapedEnvironment(MountainCarEnvironment):

    def __init__(self, env: gym.Env):
        super(MountainCarShapedEnvironment, self).__init__(env)
        self.prev_shaping = self.state[0] + 0.5

    def calculate_reward(self):
        # vel = abs(self.state[1]) * 1_000
        # if vel < 0.1:
        #     vel = 0.1
        #
        # reward = (self.state[0] - 0.5) / vel
        # if self.state[0] >= 0.5:
        #     reward = 100.0
        # return reward

        shaping = self.state[0] + 0.5
        reward = (shaping - self.prev_shaping) * abs(self.state[1]) * 10_000
        self.prev_shaping = shaping
        if self.state[0] >= 0.5:
            reward = 100
        return reward
