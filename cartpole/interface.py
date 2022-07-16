from deep_rl.agent import Terminal, Interpreter
from tensorflow import random, int32, sqrt, pow, reduce_sum


class CartpoleTerminal(Terminal):

    def __init__(self, env):
        super(CartpoleTerminal, self).__init__()
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


class CartpoleInterpreter(Interpreter):

    def __init__(self, terminal):
        super().__init__(terminal)

    def calculate_reward(self, state, preprocessed_state, reward):
        # Calculate reward from state
        reward = 0  # 0 reward for every step

        # Cart position, Cart Velocity, Pole Angle and pole angular velocity goes out of control, then give -1 reward
        if (
                state[0] >= 1 or state[0] <= -1 or
                preprocessed_state[1] >= 0.5 or preprocessed_state[1] <= -0.5 or
                preprocessed_state[2] >= 0.66 or preprocessed_state[2] <= -0.66 or
                preprocessed_state[3] >= 0.65 or preprocessed_state[3] <= -0.65
        ):
            reward = -1
            self.env_finished = True

        # If the pole is stable (pole angle within 4 degrees, pole angular velocity is within 15%,
        # cart position is within +-0.5 and cart velocity is within 50%), give reward of 1
        elif (
                (-0.33 <= preprocessed_state[2] <= 0.33) and
                (-0.15 <= preprocessed_state[3] <= 0.15) and
                (-.5 <= state[0] <= .5) and
                (-0.5 <= preprocessed_state[1] <= 0.5)
        ):
            reward = 1

        # If the agent could not balance the pole within 500 steps, end the episode
        if self.steps_taken > 500:
            self.env_finished = True

        return reward

    def state_preprocessing(self, state):
        # Scaling features
        preprocessed_state = state.copy()
        preprocessed_state[0] /= 2.4
        preprocessed_state[1] /= 2
        preprocessed_state[2] /= 0.21
        preprocessed_state[3] /= 3.5
        return preprocessed_state

    def get_randomized_action(self):
        rand = random.uniform(shape=(), maxval=1)
        action = 1
        if rand < 0.5:
            action = 0
        return action

class CartpoleShapedInterpreter(CartpoleInterpreter):

    def __init__(self, terminal):
        super().__init__(terminal)
        self.prev_shape = 0

    def calculate_reward(self, state, preprocessed_state, reward):
        shape = -sqrt(reduce_sum(pow(preprocessed_state, 2)))

        reward = shape - self.prev_shape
        self.prev_shape = shape
        return reward

