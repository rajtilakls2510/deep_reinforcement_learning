import gym, imageio, os
import cv2
from mountaincar_env_wrappers import MountainCarEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DoubleDeepQLearning
import numpy as np

env = MountainCarEnvironment(gym.make("MountainCar-v0", render_mode = "rgb_array"))

AGENT_PATH = "mountain_car_agent2"

driver_algo = DoubleDeepQLearning()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)


# Live Agent Play
agent.evaluate(mode="live", episodes=5, fps = 60)

# Put Episode in Video
# agent.evaluate(mode="video", episodes=5, path_to_video=os.path.join(AGENT_PATH, "eval"), fps=30)

