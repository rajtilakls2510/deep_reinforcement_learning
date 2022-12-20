import gym

from deep_rl.agent import GymEnvironment
from tensorflow import random, int32, zeros, concat
from tensorflow.keras.layers import Rescaling


class CarRacingEnvironment(GymEnvironment):

    def __init__(self, env: gym.Env, frame_buffer_size=3):
        super(CarRacingEnvironment, self).__init__(env)
        self.frame_buffer = zeros(shape=(96, 96, 3 * frame_buffer_size))
        self.rescale = Rescaling(scale=1.0 / 255)
        self.count = 0

    def get_randomized_action(self):
        return random.uniform(shape=(), maxval=5, dtype=int32)

    def state_preprocessing(self, state):
        self.frame_buffer = concat([self.frame_buffer[:, :, 3:], self.rescale(state)], axis=-1)
        return self.frame_buffer
