import gymnasium as gym, os
from pendulum_env_wrapper import PendulumEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import TotalRewardMetric, EpisodeLengthMetric, LiveEpisodeViewer, VideoEpisodeSaver
import tensorflow as tf
# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Wrapping the gym environment to interface with our library
env = PendulumEnvironment(gym.make("Pendulum-v1", render_mode = "rgb_array"))
AGENT_PATH = "pendulum_agent2"

driver_algo = DeepDPG()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)

# Setting up metrics for evaluation
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "eval_metric"))
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "eval_metric"))

# Live Agent Play
live_viewer = LiveEpisodeViewer(fps=60)
agent.evaluate(episodes=2, metrics=[total_reward, ep_length, live_viewer], exploration=0.0)

# Put Episodes in Video
# video_saver = VideoEpisodeSaver(os.path.join(AGENT_PATH, "eval_metric"), fps=60)
# agent.evaluate(episodes=5, metrics=[total_reward, ep_length, live_viewer, video_saver], exploration=0.0)

