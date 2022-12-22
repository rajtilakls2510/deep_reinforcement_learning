import gym
from deep_rl.analytics import Metric
from abc import ABC, abstractmethod


class DRLEnvironment(ABC):
    # Base class to handle agent interactions with the actual Environment (Might act as an interface between Agent and
    # Environment in cases where you already have an environment implemented, such as  OpenAI Gym)

    # Observes the environment and returns the state
    # Should return: state, reward, frame
    @abstractmethod
    def observe(self):
        pass

    # Calculates the reward from the state
    @abstractmethod
    def calculate_reward(self, **kwargs):
        pass

    # Preprocesses a state (Default, returns state itself)
    def preprocess_state(self, state):
        return state

    # Returns a random action
    def get_random_action(self):
        pass

    # Takes an action
    @abstractmethod
    def take_action(self, action):
        pass

    # Checks whether the episode has finished or not
    @abstractmethod
    def is_episode_finished(self):
        pass

    # Resets the environment for the next episode
    @abstractmethod
    def reset(self):
        pass

    # Closes the environment
    @abstractmethod
    def close(self):
        pass


class GymEnvironment(DRLEnvironment):
    # Implementation to interface with Gym Environments

    def __init__(self, env: gym.Env):
        self.env = env
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.preprocessed_state = None
        self.state, _ = self.env.reset()

    # Returns the current state from observation, reward, frame
    def observe(self):
        frame = self.env.render()
        self.preprocessed_state = self.preprocess_state(self.state)
        self.reward = self.calculate_reward()
        return self.preprocessed_state, self.reward, frame

    # Takes an action
    def take_action(self, action):
        self.state, self.reward, self.terminated, self.truncated, _ = self.env.step(action)

    # Defaults to returning gym reward
    def calculate_reward(self, **kwargs):
        return self.reward

    def get_random_action(self):
        return self.env.action_space.sample()

    # Checks whether the episode has finished or not
    def is_episode_finished(self):
        return self.terminated or self.truncated

    def close(self):
        self.env.close()

    def reset(self):
        self.terminated = False
        self.truncated = False
        self.state, _ = self.env.reset()


class Agent:

    def __init__(self, env: DRLEnvironment, driver_algorithm):
        self.env = env
        self.driver_algorithm = driver_algorithm
        self.driver_algorithm.set_env(env)

    def train(self, initial_episode=0, episodes=100, metrics: list[Metric] = (), **kwargs):
        for metric in metrics: metric.set_driver_algorithm(self.driver_algorithm)
        self.driver_algorithm.train(initial_episode, episodes, metrics, **kwargs)

    def evaluate(self, episodes=1, metrics: list[Metric] = (), exploration=0.0):
        for metric in metrics: metric.set_driver_algorithm(self.driver_algorithm)
        episodic_data = self.driver_algorithm.infer(episodes, metrics, exploration)
        return episodic_data

    def save(self, path=""):
        self.driver_algorithm.save(path)
        print("Agent Saved")

    def load(self, path=""):
        self.driver_algorithm.load(path)
        print("Agent Loaded")
