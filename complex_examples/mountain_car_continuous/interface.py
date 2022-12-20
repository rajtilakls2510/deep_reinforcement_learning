from deep_rl.agent import Terminal, Interpreter
import numpy as np


class MountainCarContinuousTerminal(Terminal):
    def __init__(self, env):
        super(MountainCarContinuousTerminal, self).__init__()
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


class MountainCarContinuousInterpreter(Interpreter):
    def __init__(self, terminal):
        super(MountainCarContinuousInterpreter, self).__init__(terminal)
        self.theta = 0.15
        self.mean = np.zeros(1)
        self.std_dev = float(0.2) * np.ones(1)
        self.dt = 1e-2
        self.x = np.zeros_like(self.mean)

    def get_randomized_action(self):
        # return random.normal(shape=(1,), mean=0.0, stddev=0.2, dtype=float32)
        self.x = (
                self.x
                + self.theta * (self.mean - self.x) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        return self.x

    def take_action(self, action):
        action = np.clip(action, -1, 1)
        super(MountainCarContinuousInterpreter, self).take_action(action)
