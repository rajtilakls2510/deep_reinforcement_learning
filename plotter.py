import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from deep_rl.analytics import Plotter, EpisodeLengthMetric
from deep_rl.magent_parallel import MATotalRewardMetric

AGENT_PATH = "agent"
total_reward = MATotalRewardMetric(path=os.path.join(AGENT_PATH, "train_metric"))
ep_length = EpisodeLengthMetric(path=os.path.join(AGENT_PATH, "train_metric"))

Plotter(metrics=[total_reward, ep_length]).show()