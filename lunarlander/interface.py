from deep_rl.agent import Terminal, Interpreter
from tensorflow import random, int32
import numpy as np


class LunarLanderTerminal(Terminal):
    def __init__(self, env):
        super(LunarLanderTerminal, self).__init__()
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


class LunarLanderInterpreter(Interpreter):

    def __init__(self, terminal: Terminal):
        super(LunarLanderInterpreter, self).__init__(terminal)
        self.last_action = 0

    def get_randomized_action(self):
        return random.uniform(shape=(), maxval=4, dtype=int32).numpy()
