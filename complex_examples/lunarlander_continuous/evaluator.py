import gym, os
from lunarlandercont_env_wrapper import LunarLanderContinuousEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG

interpreter = LunarLanderContinuousEnvironment(gym.make("LunarLander-v2", continuous=True, render_mode = "rgb_array"))

AGENT_PATH = "lunar_lander_cont_agent"

driver_algo = DeepDPG()
agent = Agent(interpreter, driver_algo)
agent.load(AGENT_PATH)


# Live Agent Play
agent.evaluate(mode="live", episodes=5, fps = 60)

# Put Episode in Video
# agent.evaluate(mode="video", episodes=5, path_to_video=os.path.join(AGENT_PATH, "eval"), fps=60)

