import gym, os
from cartpole_env_wrappers import CartpoleEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import TotalRewardMetric, EpisodeLengthMetric, LiveEpisodeViewer, VideoEpisodeSaver

# Wrapping the gym environment to interface with our library
env = CartpoleEnvironment(gym.make("CartPole-v1", render_mode = "rgb_array"))

AGENT_PATH = "cart_pole_agent"

driver_algo = DeepQLearning()
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
