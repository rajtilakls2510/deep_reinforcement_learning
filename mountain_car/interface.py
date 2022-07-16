from deep_rl.agent import Terminal, Interpreter
from tensorflow import random, int32


class MountainCarTerminal(Terminal):
    def __init__(self, env):
        super(MountainCarTerminal, self).__init__()
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


class MountainCarInterpreter(Interpreter):
    def __init__(self, terminal):
        super(MountainCarInterpreter, self).__init__(terminal)

    def get_randomized_action(self):
        return random.uniform(shape=(), maxval=3, dtype=int32).numpy()


class MountainCarShapedInterpreter(MountainCarInterpreter):

    def __init__(self, terminal):
        super(MountainCarShapedInterpreter, self).__init__(terminal)
        self.prev_shaping = self.terminal.observation()[1][0] + 0.5

    def calculate_reward(self, state, preprocessed_state, reward):
        # vel = abs(state[1]) * 1_000
        # if vel < 0.1:
        #     vel = 0.1
        #
        # reward = (state[0] - 0.5) / vel
        # if state[0] >= 0.5:
        #     reward = 100.0
        # return reward

        shaping = state[0] + 0.5
        reward = (shaping - self.prev_shaping) * abs(state[1]) * 10_000
        self.prev_shaping = shaping
        if state[0] >= 0.5:
            reward = 100
        return reward
