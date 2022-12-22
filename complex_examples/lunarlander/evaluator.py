import gym, os
from lunarlander_env_wrappers import LunarLanderEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning

env = LunarLanderEnvironment(gym.make("LunarLander-v2", render_mode = "rgb_array"))

AGENT_PATH = "lunar_lander_agent"

driver_algo = DeepQLearning()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)


# Live Agent Play
agent.evaluate(mode="live", episodes=5, fps = 60)

# Put Episode in Video
# agent.evaluate(mode="video", episodes=5, path_to_video=os.path.join(AGENT_PATH, "eval"), fps=30)

