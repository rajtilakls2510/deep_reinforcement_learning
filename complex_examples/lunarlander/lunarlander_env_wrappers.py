import gym
from deep_rl.agent import GymEnvironment
from tensorflow import random, int32, zeros
from tensorflow.keras.layers import Rescaling, Resizing


class LunarLanderEnvironment(GymEnvironment):

    def get_random_action(self):
        return random.uniform(shape=(), maxval=4, dtype=int32)


class LunarLanderImageEnvironment(LunarLanderEnvironment):
    def __init__(self, env: gym.Env, frame_buffer_size=5, preprocessed_frame_shape=(80,80,3)):
        super(LunarLanderImageEnvironment, self).__init__(env)
        self.frame_buffer = [zeros(shape=preprocessed_frame_shape).numpy() for _ in range(frame_buffer_size)]
        self.frame_buffer_size = frame_buffer_size
        self.rescale = Rescaling(scale=1.0/255)
        self.resize = Resizing(width=preprocessed_frame_shape[0], height=preprocessed_frame_shape[1])

    def preprocess_state(self, frame):
        if len(self.frame_buffer) >= self.frame_buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(self.resize(self.rescale(frame)).numpy())
        return self.frame_buffer
