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

    def state_preprocessing(self, state):
        return state

    def calculate_reward(self, state, preprocessed_state, reward):
        # reward = (state[0] + 0.3)/1.8
        reward = -1

        if state[0] >= .5:
            reward = 1

        if self.steps_taken > 200:
            self.env_finished = True

        return reward

    def get_randomized_action(self):
        return random.uniform(shape=(), maxval=2, dtype=int32).numpy()
