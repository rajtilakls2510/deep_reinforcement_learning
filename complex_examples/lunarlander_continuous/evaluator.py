import gym, os
from lunarlandercont_env_wrapper import LunarLanderContinuousEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import TotalRewardMetric, EpisodeLengthMetric, LiveEpisodeViewer, VideoEpisodeSaver

# Wrapping the gym environment to interface with our library
env = LunarLanderContinuousEnvironment(gym.make("LunarLander-v2", continuous=True, render_mode="rgb_array"))

AGENT_PATH = "lunar_lander_cont_agent"

driver_algo = DeepDPG()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)

# Setting up metrics for evaluation
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "eval_metric"))
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "eval_metric"))

# Live Agent Play
live_viewer = LiveEpisodeViewer(fps=60)
agent.evaluate(episodes=5, metrics=[total_reward, ep_length, live_viewer], exploration=0.0)

# Put Episodes in Video
# video_saver = VideoEpisodeSaver(os.path.join(AGENT_PATH, "eval_metric"), fps=60)
# agent.evaluate(episodes=5, metrics=[total_reward, ep_length, video_saver], exploration=0.0)
