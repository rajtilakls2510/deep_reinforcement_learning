from deep_rl.agent import Terminal, Interpreter
from tensorflow import random, int32, zeros, concat
from tensorflow.keras.layers import Rescaling


class CarRacingTerminal(Terminal):
    def __init__(self, env):
        super(CarRacingTerminal, self).__init__()
        self.env = env
        self.env_finished = False
        self.reward = 0
        self.state = self.env.reset()

    def observation(self):
        return self.env.render(mode="rgb_array"), self.state, self.env_finished, self.reward

    def action(self, action):
        self.state, self.reward, self.env_finished, _ = self.env.step(action)

    def close(self):
        self.env.close()

    def reset(self):
        self.env_finished = False
        self.state = self.env.reset()


class CarRacingInterpreter(Interpreter):

    def __init__(self, terminal: Terminal, frame_buffer_size=3):
        super(CarRacingInterpreter, self).__init__(terminal)
        self.frame_buffer = zeros(shape=(96, 96, 3 * frame_buffer_size))
        self.rescale = Rescaling(scale=1.0 / 255)
        self.count = 0

    def get_randomized_action(self):
        return random.uniform(shape=(), maxval=5, dtype=int32)

    def state_preprocessing(self, state):
        self.frame_buffer = concat([self.frame_buffer[:,:,3:], self.rescale(state)], axis=-1)
        return self.frame_buffer
