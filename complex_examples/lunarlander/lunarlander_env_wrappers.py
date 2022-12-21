from deep_rl.agent import GymEnvironment
from tensorflow import random, int32, zeros


class LunarLanderEnvironment(GymEnvironment):

    def get_random_action(self):
        return random.uniform(shape=(), maxval=4, dtype=int32)

