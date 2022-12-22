import gym, os
from mountaincarcont_env_wrappers import MountainCarContinuousEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG

env = MountainCarContinuousEnvironment(gym.make("MountainCarContinuous-v0", render_mode="rgb_array"))
AGENT_PATH = "mountain_car_cont_agent"

driver_algo = DeepDPG()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)

# Live Agent Play
agent.evaluate(mode="live", episodes=5, fps=60)

# Put Episode in Video
# agent.evaluate(mode="video", episodes=5, path_to_video=os.path.join(AGENT_PATH, "eval"), fps=60)
