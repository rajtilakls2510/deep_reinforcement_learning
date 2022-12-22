import pandas as pd
import os, cv2, imageio
import tensorflow as tf


class Metric:

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


# Tracks the length of the episode
class EpisodeLengthMetric(Metric):

    def __init__(self, path=""):
        super(EpisodeLengthMetric, self).__init__(path)
        self.name = "Episode Length"
        self.episodic_data = {"episode": [], "length": []}
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


# Tracks the total reward in an episode
class TotalRewardMetric(Metric):

    def __init__(self, path=""):
        super(TotalRewardMetric, self).__init__(path)
        self.name = "Total Reward"
        self.episodic_data = {"episode": [], "total_reward": []}
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


# Tracks the Average Q value of some random states drawn at the beginning of the task
class AverageQMetric(Metric):

    def __init__(self, path=""):
        super(AverageQMetric, self).__init__(path)
        self.name = "Avg Q"
        self.episodic_data = {"episode": [], "avg_q": []}
        self.random_states = tf.constant([], dtype=tf.float32)

    def on_task_begin(self, data=None):
        self.load()
        if self.random_states.shape[0] == 0:
            self.random_states = self.driver_algorithm.get_random_states()

    def on_episode_end(self, data=None):
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["avg_q"].append(tf.reduce_mean(self.driver_algorithm.get_values(self.random_states)).numpy())
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


# Tracks the Exploration (epsilon) value of every episode
class ExplorationTrackerMetric(Metric):

    def __init__(self, path=""):
        super(ExplorationTrackerMetric, self).__init__(path)
        self.name = "Exploration Tracker"
        self.episodic_data = {"episode": [], "exploration": []}

    def on_task_begin(self, data=None):
        self.load()

    def on_episode_end(self, data=None):
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["exploration"].append(data["exploration"])
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


# Used to view the live interaction with environment
class LiveEpisodeViewer(Metric):

    def __init__(self, path="", fps=30):
        super(LiveEpisodeViewer, self).__init__(path)
        self.name = "Live Episode"
        self.winname = ""
        self.fps = fps

    def on_episode_begin(self, data=None):
        print("\nEpisode:", str(data["episode"] + 1))
        self.winname = "Episode: " + str(data["episode"] + 1)
        cv2.namedWindow(self.winname)  # Create a named window
        cv2.moveWindow(self.winname, 50, 50)  # Move it to (40,30)

    def on_episode_step(self, data=None):
        cv2.imshow(self.winname, cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB))
        cv2.waitKey(1000 // self.fps)

    def on_episode_end(self, data=None):
        cv2.destroyWindow(self.winname)


# Used to store agent's interaction with environment as a video
class VideoEpisodeSaver(Metric):

    def __init__(self, path="", fps=30):
        super(VideoEpisodeSaver, self).__init__(path)
        self.name = "Video Saver"
        self.fps = fps
        self.writer = None

    def on_task_begin(self, data=None):
        os.makedirs(os.path.join(self.path), exist_ok=True)

    def on_episode_begin(self, data=None):
        self.writer = imageio.get_writer(os.path.join(self.path, "vid_" + str(data["episode"] + 1) + ".mp4"),
                                         fps=self.fps)

    def on_episode_step(self, data=None):
        self.writer.append_data(data["frame"])

    def on_episode_end(self, data=None):
        self.writer.close()

# class AvgTotalReward(Metric):
#
#     def __init__(self, path=""):
#         super(AvgTotalReward, self).__init__(path)
#         self.episodic_data = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}
#         self.current_episode = []
#         self.random_states = tf.constant([], dtype=tf.float32)
#
#     def on_task_begin(self, data=None):
#         print("Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
#         if self.random_states.shape[0] == 0:
#             self.random_states = self.driver_algorithm.get_random_states()
#
#     def on_episode_begin(self, data=None):
#         self.current_episode = []
#
#     # data: {"action_value":0, "action":0, "reward": 0, "explored":False, "next_action_value":0, "next_action":0}
#     def on_episode_step(self, data=None):
#         self.current_episode.append(data)
#
#     # data: {"episode":0, "exploration":0.0}
#     def on_episode_end(self, data=None):
#         total_reward = 0
#         for step in self.current_episode:
#             total_reward += step["reward"]
#         self.episodic_data["episode"].append(data["episode"])
#         self.episodic_data["length"].append(len(self.current_episode))
#         self.episodic_data["total_reward"].append(total_reward)
#         self.episodic_data["avg_q"].append(tf.reduce_mean(self.driver_algorithm.get_values(self.random_states)).numpy())
#         self.episodic_data["exploration"].append(data["exploration"])
#
#     # def on_task_end(self, data=None):
#
#     def save(self):
#         tf.io.write_file(os.path.join(self.path, "random_states.tfw"), tf.io.serialize_tensor(self.random_states))
#         pd.DataFrame(self.episodic_data).to_csv(os.path.join(self.path, "episodic_data.csv"), index=False)
#
#     def load(self):
#         try:
#             self.episodic_data = pd.read_csv(os.path.join(self.path, "episodic_data.csv")).to_dict('list')
#             self.random_states = tf.io.parse_tensor(tf.io.read_file(os.path.join(self.path, "random_states.tfw")),
#                                                     tf.float32)
#         except:
#             print("No Random States found")
