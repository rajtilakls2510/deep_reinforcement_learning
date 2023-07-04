import gymnasium as gym, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from ant_env_wrappers import AntEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG, TD3
from deep_rl.analytics import TotalRewardMetric, EpisodeLengthMetric, LiveEpisodeViewer, VideoEpisodeSaver
import tensorflow as tf

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Wrapping the gym environment to interface with our library
env = AntEnvironment(gym.make("Hopper-v4", render_mode="rgb_array", terminate_when_unhealthy=False))
AGENT_PATH = "hopper_agent"

driver_algo = TD3()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)

# Setting up metrics for evaluation
# total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "eval_metric"))
# ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "eval_metric"))

# Live Agent Play
live_viewer = LiveEpisodeViewer(fps=125)
agent.evaluate(initial_episode=0, episodes=2, metrics=[live_viewer], exploration=0.0, rendering=True)

# Put Episodes in Video
# video_saver = VideoEpisodeSaver(os.path.join(AGENT_PATH, "eval_metric"), fps=20)
# agent.evaluate(initial_episode=0, episodes=5, metrics=[video_saver], exploration=0.2, rendering=True)
env.close()
