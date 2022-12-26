import os
# Set this to not use Tensorflow-GPU for just for plotting (It unnecessarily uses up free GPU memory)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from deep_rl.analytics import Plotter, EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, AbsoluteValueErrorMetric
# =================== Plotting the live metrics of an Agent ==================

AGENT_PATH = "mountain_car_cont_agent"

# Setting up metric to live track
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))

# Class to plot the metrics as live graphs
plotter = Plotter(metrics = [total_reward, ep_length, avg_q, value_error])
plotter.show()