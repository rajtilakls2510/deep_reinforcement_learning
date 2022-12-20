from deep_rl.agent import GymEnvironment
from tensorflow import random, int32, sqrt, pow, reduce_sum


class CartpoleEnvironment(GymEnvironment):

    def calculate_reward(self):
        # Calculate reward from state
        reward = 0  # 0 reward for every step

        # Cart position, Cart Velocity, Pole Angle and pole angular velocity goes out of control, then give -1 reward
        if (
                self.state[0] >= 1 or self.state[0] <= -1 or
                self.preprocessed_state[1] >= 0.5 or self.preprocessed_state[1] <= -0.5 or
                self.preprocessed_state[2] >= 0.66 or self.preprocessed_state[2] <= -0.66 or
                self.preprocessed_state[3] >= 0.65 or self.preprocessed_state[3] <= -0.65
        ):
            reward = -1
            self.env_finished = True

        # If the pole is stable (pole angle within 4 degrees, pole angular velocity is within 15%,
        # cart position is within +-0.5 and cart velocity is within 50%), give reward of 1
        elif (
                (-0.33 <= self.preprocessed_state[2] <= 0.33) and
                (-0.15 <= self.preprocessed_state[3] <= 0.15) and
                (-.5 <= self.state[0] <= .5) and
                (-0.5 <= self.preprocessed_state[1] <= 0.5)
        ):
            reward = 1

        # If the agent could not balance the pole within 500 steps, end the episode
        if self.steps_taken > 500:
            self.env_finished = True

        return reward

    def preprocess_state(self, state):
        # Scaling features
        preprocessed_state = state.copy()
        preprocessed_state[0] /= 2.4
        preprocessed_state[1] /= 2
        preprocessed_state[2] /= 0.21
        preprocessed_state[3] /= 3.5
        return preprocessed_state

    def get_random_action(self):
        return random.uniform(shape=(), maxval=2, dtype=int32)


class CartpoleShapedEnvironment(CartpoleEnvironment):

    def __init__(self, env):
        super().__init__(env)
        self.prev_shape = 0

    def calculate_reward(self):
        shape = -sqrt(reduce_sum(pow(self.preprocessed_state, 2)))

        reward = shape - self.prev_shape
        self.prev_shape = shape
        return reward
