import pandas as pd
from abc import ABC
import os, cv2, imageio, math
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cycler import cycler
from datetime import datetime


class Metric(ABC):
    # Base class to implement a metric for an algorithm
    # Description: This class acts as a callback where methods of this class are called during training and
    #           evaluation runs. You can use this class to keep track of different things during training,
    #           evaluation and so on. You can also use this class to see the live interactions of the agent
    #           with the environment or save the interaction frames as a video. Implementations of such are
    #           provided through the LiveEpisodeViewer and VideoEpisodeSaver subclasses.

    # Preferably takes a path to save this metric
    def __init__(self, path=""):
        self.path = path
        self.driver_algorithm = None
        self.name = ""

    # Driver Algorithm is provided such that any member of this class can be accessed for calculation
    def set_driver_algorithm(self, driver_algorithm):
        self.driver_algorithm = driver_algorithm

    # Called when this particular task begins (typically, training or evaluation tasks)
    def on_task_begin(self, data=None):
        pass

    # Called before an episode is about to begin
    def on_episode_begin(self, data=None):
        pass

    # Called after every step of an episode
    def on_episode_step(self, data=None):
        pass

    # Called after an episode has ended
    def on_episode_end(self, data=None):
        pass

    # Called when this particular task begins (typically, training or evaluation tasks)
    def on_task_end(self, data=None):
        pass

    # Saves the metric
    def save(self):
        pass

    # Loads the metric
    def load(self):
        pass

    # Returns data (as a dictionary) that can be plotted on graph,
    # the first key will be used as x-axis and the rest as the y-axes
    def get_plot_data(self):
        pass


class EpisodeLengthMetric(Metric):
    # Tracks the length of the episode

    def __init__(self, path=""):
        super(EpisodeLengthMetric, self).__init__(path)
        self.name = "Episode Length"
        self.episodic_data = {"episode": [], "length": [], "step": []}
        self.length = 0

    def on_task_begin(self, data=None):
        self.load()

    def on_episode_begin(self, data=None):
        self.length = 0

    def on_episode_step(self, data=None):
        self.length += 1

    def on_episode_end(self, data=None):
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["length"].append(self.length)
        self.episodic_data["step"].append(data["step"])
        self.save()

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        df = pd.DataFrame(self.episodic_data)
        df.drop_duplicates(subset=["episode"], keep="last", inplace=True)
        df.to_csv(os.path.join(self.path, self.name + ".csv"), index=False)

    def load(self):
        try:
            self.episodic_data = pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
            print(self.name + " : Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
        except:
            print(self.name + " : No Previous Data Found.")

    def get_plot_data(self):
        try:
            return pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
        except:
            return {"episode": [], "length": [], "step": []}


class TotalRewardMetric(Metric):
    # Tracks the total reward in an episode

    def __init__(self, path=""):
        super(TotalRewardMetric, self).__init__(path)
        self.name = "Total Reward"
        self.episodic_data = {"episode": [], "total_reward": [], "step": []}
        self.total_reward = 0

    def on_task_begin(self, data=None):
        self.load()

    def on_episode_begin(self, data=None):
        self.total_reward = 0

    def on_episode_step(self, data=None):
        self.total_reward += data["reward"]

    def on_episode_end(self, data=None):
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["total_reward"].append(self.total_reward)
        self.episodic_data["step"].append(data["step"])
        self.save()

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        df = pd.DataFrame(self.episodic_data)
        df.drop_duplicates(subset=["episode"], keep="last", inplace=True)
        df.to_csv(os.path.join(self.path, self.name + ".csv"), index=False)

    def load(self):
        try:
            self.episodic_data = pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
            print(self.name + " : Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
        except:
            print(self.name + " : No Previous Data Found.")

    def get_plot_data(self):
        try:
            return pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
        except:
            return {"episode": [], "total_reward": [], "step": []}


class AverageQMetric(Metric):
    # Tracks the Average Q value of some random states drawn at the beginning of the task

    def __init__(self, path=""):
        super(AverageQMetric, self).__init__(path)
        self.name = "Avg Q"
        self.episodic_data = {"episode": [], "avg_q": [], "step": []}
        self.random_states = tf.constant([], dtype=tf.float32)

    def on_task_begin(self, data=None):
        self.load()
        if self.random_states.shape[0] == 0:
            self.random_states = self.driver_algorithm.get_random_states()

    def on_episode_end(self, data=None):
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["avg_q"].append(tf.reduce_mean(self.driver_algorithm.get_values(self.random_states)).numpy())
        self.episodic_data["step"].append(data["step"])
        self.save()

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        tf.io.write_file(os.path.join(self.path, "random_states.tfw"), tf.io.serialize_tensor(self.random_states))
        df = pd.DataFrame(self.episodic_data)
        df.drop_duplicates(subset=["episode"], keep="last", inplace=True)
        df.to_csv(os.path.join(self.path, self.name + ".csv"), index=False)

    def load(self):
        try:
            self.episodic_data = pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
            self.random_states = tf.io.parse_tensor(tf.io.read_file(os.path.join(self.path, "random_states.tfw")),
                                                    tf.float32)
            print(self.name + " : Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
        except:
            print(self.name + " : No Random States found")

    def get_plot_data(self):
        try:
            return pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
        except:
            return {"episode": [], "avg_q": [], "step": []}


class ExplorationTrackerMetric(Metric):
    # Tracks the Exploration (epsilon) value of every episode

    def __init__(self, path=""):
        super(ExplorationTrackerMetric, self).__init__(path)
        self.name = "Exploration Tracker"
        self.episodic_data = {"episode": [], "exploration": [], "step": []}

    def on_task_begin(self, data=None):
        self.load()

    def on_episode_end(self, data=None):
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["exploration"].append(data["exploration"])
        self.episodic_data["step"].append(data["step"])
        self.save()

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        df = pd.DataFrame(self.episodic_data)
        df.drop_duplicates(subset=["episode"], keep="last", inplace=True)
        df.to_csv(os.path.join(self.path, self.name + ".csv"), index=False)

    def load(self):
        try:
            self.episodic_data = pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
            print(self.name + " : Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
        except:
            print(self.name + " : No Previous Data Found.")

    def get_plot_data(self):
        try:
            return pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
        except:
            return {"episode": [], "exploration": [], "step": []}


class AbsoluteValueErrorMetric(Metric):
    # Tracks the total value error against the actual return per episode

    def __init__(self, path=""):
        super(AbsoluteValueErrorMetric, self).__init__(path)
        self.name = "Absolute Value Error"
        self.episodic_data = {"episode": [], "absolute error": [], "step": []}
        self.values = []
        self.rewards = []

    def on_task_begin(self, data=None):
        self.load()

    def on_episode_begin(self, data=None):
        self.values = []
        self.rewards = []

    def on_episode_step(self, data=None):
        self.values.insert(0, data["action_value"])
        self.rewards.insert(0, data["reward"])

    def on_episode_end(self, data=None):
        total_error = 0
        ret = 0
        for value, reward in zip(self.values, self.rewards):
            ret = ret * self.driver_algorithm.discount_factor.numpy() + reward
            total_error += abs(ret - value)
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["absolute error"].append(total_error)
        self.episodic_data["step"].append(data["step"])
        self.save()

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        df = pd.DataFrame(self.episodic_data)
        df.drop_duplicates(subset=["episode"], keep="last", inplace=True)
        df.to_csv(os.path.join(self.path, self.name + ".csv"), index=False)

    def load(self):
        try:
            self.episodic_data = pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
            print(self.name + " : Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
        except:
            print(self.name + " : No Previous Data Found.")

    def get_plot_data(self):
        try:
            return pd.read_csv(os.path.join(self.path, self.name + ".csv")).to_dict('list')
        except:
            return {"episode": [], "absolute error": [], "step": []}


class LiveEpisodeViewer(Metric):
    # Used to view the live interaction with environment

    def __init__(self, path="", fps=30):
        super(LiveEpisodeViewer, self).__init__(path)
        self.name = "Live Episode"
        self.winname = ""
        self.fps = fps

    def on_episode_begin(self, data=None):
        print("Episode:", str(data["episode"] + 1))
        self.winname = "Episode: " + str(data["episode"] + 1)
        cv2.namedWindow(self.winname)
        cv2.moveWindow(self.winname, 50, 50)

    def on_episode_step(self, data=None):
        cv2.imshow(self.winname, cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB))
        cv2.waitKey(1000 // self.fps)

    def on_episode_end(self, data=None):
        cv2.destroyWindow(self.winname)


class VideoEpisodeSaver(Metric):
    # Used to store agent's interaction with environment as a video

    def __init__(self, path="", fps=30):
        super(VideoEpisodeSaver, self).__init__(path)
        self.name = "Video Saver"
        self.fps = fps
        self.writer = None

    def on_task_begin(self, data=None):
        os.makedirs(os.path.join(self.path), exist_ok=True)

    def on_episode_begin(self, data=None):
        self.writer = imageio.get_writer(os.path.join(self.path, "Episode_" + str(data["episode"] + 1) + ".mp4"),
                                         fps=self.fps)

    def on_episode_step(self, data=None):
        self.writer.append_data(data["frame"])

    def on_episode_end(self, data=None):
        self.writer.close()


class SaveAgent(Metric):

    def __init__(self, path="", save_after_episodes=1):
        super(SaveAgent, self).__init__(path)
        self.name = "Save Agent"
        self.current = 0
        self.save_after = save_after_episodes

    def on_episode_end(self, data=None):
        self.current += 1
        if self.current % self.save_after == 0:
            episode = data["episode"]
            os.makedirs(os.path.join(self.path, f"Episode_{episode+1} {datetime.now().strftime('%Y_%m_%d %H_%M_%S')}"), exist_ok=True)
            self.driver_algorithm.save(os.path.join(self.path, f"Episode_{episode+1} {datetime.now().strftime('%Y_%m_%d %H_%M_%S')}"))


class Plotter:
    # This class is used to plot the metrics that are/were being tracked during training or evaluation.
    # Simply initialize this class with the list of metrics that you want to track with appropriate paths
    # and call the show() method. This will bring up a matplotlib figure where you will be able to see graphs
    # for all the metrics that are being tracked.

    def __init__(self, metrics: list[Metric], frequency=5000, smoothing=0.8, name="Figure1"):
        self.frequency = frequency
        self.metrics = metrics
        self.cols = 3
        self.rows = math.ceil(len(metrics) / self.cols)
        self.smoothingWeight = smoothing
        self.name = name

        # Setting up plot styles
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

        plt.rcParams["lines.color"] = "#F8F8F2"
        plt.rcParams["patch.edgecolor"] = "#F8F8F2"

        plt.rcParams["text.color"] = "#F8F8F2"

        plt.rcParams["axes.facecolor"] = "#282A36"
        plt.rcParams["axes.edgecolor"] = "#F8F8F2"
        plt.rcParams["axes.labelcolor"] = "#F8F8F2"

        plt.rcParams["axes.prop_cycle"] = cycler('color',
                                                 ['#8be9fd', '#ff79c6', '#50fa7b', '#bd93f9', '#ffb86c', '#ff5555',
                                                  '#f1fa8c', '#6272a4'])

        plt.rcParams["xtick.color"] = "#F8F8F2"
        plt.rcParams["ytick.color"] = "#F8F8F2"

        plt.rcParams["legend.framealpha"] = 0.9
        plt.rcParams["legend.edgecolor"] = "#44475A"

        plt.rcParams["grid.color"] = "#F8F8F2"

        plt.rcParams["figure.facecolor"] = "#383A59"
        plt.rcParams["figure.edgecolor"] = "#383A59"

        plt.rcParams["savefig.facecolor"] = "#383A59"
        plt.rcParams["savefig.edgecolor"] = "#383A59"

        # Boxplots
        plt.rcParams["boxplot.boxprops.color"] = "F8F8F2"
        plt.rcParams["boxplot.capprops.color"] = "F8F8F2"
        plt.rcParams["boxplot.flierprops.color"] = "F8F8F2"
        plt.rcParams["boxplot.flierprops.markeredgecolor"] = "F8F8F2"
        plt.rcParams["boxplot.whiskerprops.color"] = "F8F8F2"

        self.colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        self.colors = self.colors * (len(self.metrics) // len(self.colors) + 1)
        self.fig, self.axes = plt.subplots(nrows=self.rows, ncols=self.cols, num=self.name)
        if len(self.metrics) % self.cols > 0:
            if self.rows>1:
                for i in range(self.cols - len(self.metrics) % self.cols):
                    self.fig.delaxes(self.axes[-1][self.cols - i - 1])
            else:
                for i in range(self.cols - len(self.metrics) % self.cols):
                    self.fig.delaxes(self.axes[self.cols - i - 1])
        plt.subplots_adjust(left=0.05,
                            bottom=0.1,
                            right=0.95,
                            top=0.95,
                            wspace=0.2,
                            hspace=0.3)

    def plot(self, f):
        data = [metric.get_plot_data() for metric in self.metrics]

        for ax, d, metric, color in zip(self.axes.flat, data, self.metrics, self.colors):
            xlabel, ylabel = d.keys()
            smoothed = []
            try:
                last = d[ylabel][0]
                for smoothee in d[ylabel]:
                    last = last * self.smoothingWeight + (1 - self.smoothingWeight) * smoothee
                    smoothed.append(last)
            except:
                pass
            ax.clear()
            ax.plot(d[xlabel], d[ylabel], color=color, linewidth=0.5, alpha=0.25)  # Original Plot
            ax.plot(d[xlabel], smoothed, color=color, linewidth=1)  # Smoothed Plot
            ax.set_title(metric.name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(visible=True, linewidth=0.05)

    def show(self, block=True):
        anim = FuncAnimation(self.fig, self.plot, interval=self.frequency)
        plt.show(block=block)


class TrainStepPlotter(Plotter):
    def __init__(self, metrics: list[Metric], frequency=5000, smoothing=0.8, name="Figure1"):
        super().__init__(metrics, frequency, smoothing, name)

    def plot(self, f):
        data = [metric.get_plot_data() for metric in self.metrics]

        for ax, d, metric, color in zip(self.axes.flat, data, self.metrics, self.colors):
            _, ylabel, xlabel = d.keys()
            smoothed = []
            try:
                last = d[ylabel][0]
                for smoothee in d[ylabel]:
                    last = last * self.smoothingWeight + (1 - self.smoothingWeight) * smoothee
                    smoothed.append(last)
            except:
                pass
            ax.clear()
            ax.plot(d[xlabel], d[ylabel], color=color, linewidth=0.5, alpha=0.25)  # Original Plot
            ax.plot(d[xlabel], smoothed, color=color, linewidth=1)  # Smoothed Plot
            ax.set_title(metric.name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(visible=True, linewidth=0.05)