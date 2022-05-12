# Architecture Revamp:
#  -
#  - Write some inference metric
#  -
#  -
#  -
from deep_rl.analytics import Metric


class Terminal:
    # Base class to seamlessly interface with any kind of environment (even non-gym)

    # Returns the current observation from the environment
    # Returns: frame, state, env_finished or not, reward
    def observation(self):
        return None, None, True, 0

    # Takes an action on the environment
    def action(self, action):
        pass

    # Closes the environment (might be optional)
    def close(self):
        pass

    # Resets the environment so that new episode can begin
    def reset(self):
        pass


class Interpreter:
    # Base class to handle agent interactions with the Terminal class

    # Constructor takes any kind of Terminal
    def __init__(self, terminal: Terminal):
        self.terminal = terminal
        self.steps_taken = 0
        self.env_finished = False

    # Returns the current state from observation, reward, frame
    def observe(self):
        frame, state, self.env_finished, reward = self.terminal.observation()
        self.steps_taken += 1
        preprocessed_state = self.state_preprocessing(state)
        reward = self.calculate_reward(state, preprocessed_state, reward)
        if self.env_finished:
            self.steps_taken = 0
        return preprocessed_state, reward, frame

    # Calculates and returns the reward for the current state (if no manual reward calculation is needed,
    # reward observed from the environment is returned)
    # For convenience, both the state and its preprocessed version is given to the function
    def calculate_reward(self, state, preprocessed_state, reward):
        return reward  # Calculates the reward based on the state

    # Preprocesses a state
    def state_preprocessing(self, state):
        pass

    # Returns a random action
    def get_randomized_action(self):
        pass

    # Takes an action
    def take_action(self, action):
        self.terminal.action(action)

    # Checks whether the episode has finished or not
    def is_episode_finished(self):
        return self.env_finished

    def close(self):
        self.terminal.close()

    def reset(self):
        self.terminal.reset()


class Agent:

    def __init__(self, interpreter: Interpreter, driver_algorithm):
        self.interpreter = interpreter
        self.driver_algorithm = driver_algorithm
        self.driver_algorithm.set_interpreter(interpreter)

    def train(self, initial_episode=0, episodes=100, metric=None):
        print("Starting Training")
        if metric is None:
            metric = Metric()
        metric.set_driver_algorithm(self.driver_algorithm)
        self.driver_algorithm.train(initial_episode, episodes, metric)
        print("Training Complete")

    def infer(self, episodes=1, metric=None, exploration=0.0):
        if metric is None:
            metric = Metric()
        metric.set_driver_algorithm(self.driver_algorithm)
        episodic_data = self.driver_algorithm.infer(episodes, metric, exploration)
        return episodic_data

    def save(self, path=""):
        self.driver_algorithm.save(path)
        print("Agent Saved")

    def load(self, path=""):
        self.driver_algorithm.load(path)
        print("Agent Loaded")
