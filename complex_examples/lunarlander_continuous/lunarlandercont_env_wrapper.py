from deep_rl.agent import GymEnvironment
import numpy as np


# Creating Custom Gym Environment wrapper to change the random action function
class LunarLanderContinuousEnvironment(GymEnvironment):

    def __init__(self, env):
        super(LunarLanderContinuousEnvironment, self).__init__(env)
        self.theta = 0.15
        self.mean = np.zeros(2)
        self.std_dev = float(0.2) * np.ones(2)
        self.dt = 1e-2
        self.x = np.zeros_like(self.mean)

    def observe(self):
        # frame = self.env.render()
        self.preprocessed_state = self.preprocess_state(self.state)
        self.reward = self.calculate_reward()
        return self.preprocessed_state, self.reward, None

    def get_randomized_action(self):
        self.x = (
                self.x
                + self.theta * (self.mean - self.x) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        return self.x

    def take_action(self, action):
        action = np.clip(action, -1, 1)
        super(LunarLanderContinuousEnvironment, self).take_action(action)
