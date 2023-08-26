import os
# Set this to not use Tensorflow-GPU for just for plotting (It unnecessarily uses up free GPU memory)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from deep_rl.analytics import Plotter, EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, AbsoluteValueErrorMetric,TrainStepPlotter
# =================== Plotting the live metrics of an Agent ==================

AGENT_PATH = "ant_agent"

# Setting up metric to live track
ep_length_train = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward_train = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "eval_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "eval_metric"))
# avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
# value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))

# Class to plot the metrics as live graphs
plotter = TrainStepPlotter(metrics = [ep_length_train, total_reward_train, total_reward, ep_length], name="TD3")
plotter.show()