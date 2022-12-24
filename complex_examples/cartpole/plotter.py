from deep_rl.analytics import Plotter, EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, ExplorationTrackerMetric, AbsoluteValueErrorMetric
import os

# =================== Plotting the live metrics of an Agent ==================

AGENT_PATH = "cart_pole_agent"

# Setting up metric to live track
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
exp_tracker = ExplorationTrackerMetric(os.path.join(AGENT_PATH, "train_metric"))
value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))

# Class to plot the metrics as live graphs
plotter = Plotter(metrics = [total_reward, ep_length, avg_q, exp_tracker, value_error])
plotter.show()