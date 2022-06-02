from deep_rl.agent import Terminal, Interpreter
from tensorflow import random, int32, zeros
from tensorflow.keras.layers import Rescaling, Resizing


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
        self.state = self.env.reset(seed=int(random.uniform(shape=(), minval=0, maxval=1000, dtype=int32).numpy()))


class LunarLanderInterpreter(Interpreter):

    def __init__(self, terminal: Terminal):
        super(LunarLanderInterpreter, self).__init__(terminal)

    def get_randomized_action(self):
        return random.uniform(shape=(), maxval=4, dtype=int32)


class LunarLanderImageInterpreter(LunarLanderInterpreter):
    def __init__(self, terminal: Terminal, frame_buffer_size=5, preprocessed_frame_shape=(80,80,3)):
        super(LunarLanderImageInterpreter, self).__init__(terminal)
        self.frame_buffer = [zeros(shape=preprocessed_frame_shape).numpy() for _ in range(frame_buffer_size)]
        self.frame_buffer_size = frame_buffer_size
        self.rescale = Rescaling(scale=1.0/255)
        self.resize = Resizing(width=preprocessed_frame_shape[0], height=preprocessed_frame_shape[1])

    def observe(self):
        frame, state, self.env_finished, reward = self.terminal.observation()
        self.steps_taken += 1
        preprocessed_state = self.state_preprocessing(frame)
        reward = self.calculate_reward(state, preprocessed_state, reward)
        if self.env_finished:
            self.steps_taken = 0
        return preprocessed_state, reward, frame

    def state_preprocessing(self, frame):
        if len(self.frame_buffer) >= self.frame_buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(self.resize(self.rescale(frame)).numpy())
        return self.frame_buffer
