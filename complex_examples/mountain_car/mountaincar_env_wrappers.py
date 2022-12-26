from deep_rl.agent import GymEnvironment


class MountainCarEnvironment(GymEnvironment):

    def calculate_reward(self):
        vel = abs(self.state[1]) * 1_000
        if vel < 0.1:
            vel = 0.1

        reward = (self.state[0] - 0.5) / vel
        if self.state[0] >= 0.5:
            reward = 100.0
        return reward

