from deep_rl.analytics import Plotter, EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, ExplorationTrackerMetric, RegretMetric
import os

AGENT_PATH = "cart_pole_agent3"

ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
exp_tracker = ExplorationTrackerMetric(os.path.join(AGENT_PATH, "train_metric"))
regret = RegretMetric(os.path.join(AGENT_PATH, "train_metric"))

plotter = Plotter(metrics = [total_reward, ep_length, avg_q, exp_tracker, regret])
plotter.show()