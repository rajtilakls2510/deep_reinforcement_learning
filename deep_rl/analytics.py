import pandas as pd
import os
import tensorflow as tf


class Metric:

    def __init__(self, path=""):
        self.path = path
        self.driver_algorithm = None

    def set_driver_algorithm(self, driver_algorithm):
        self.driver_algorithm = driver_algorithm

    def on_task_begin(self, data=None):
        pass

    def on_episode_begin(self, data=None):
        pass

    def on_episode_step(self, data=None):
        pass

    def on_episode_end(self, data=None):
        pass

    def on_task_end(self, data=None):
        pass

    def save(self):
        pass

    def load(self):
        pass


class AvgTotalReward(Metric):

    def __init__(self, path=""):
        super(AvgTotalReward, self).__init__(path)
        self.episodic_data = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}
        self.current_episode = []
        self.random_states = tf.constant([], dtype=tf.float32)

    def on_task_begin(self, data=None):
        print("Found " + str(len(self.episodic_data["episode"])) + " episode(s)")
        if self.random_states.shape[0] == 0:
            self.random_states = self.driver_algorithm.get_random_states()

    def on_episode_begin(self, data=None):
        self.current_episode = []

    # data: {"action_value":0, "action":0, "reward": 0, "explored":False, "next_action_value":0, "next_action":0}
    def on_episode_step(self, data=None):
        self.current_episode.append(data)

    # data: {"episode":0, "exploration":0.0}
    def on_episode_end(self, data=None):
        total_reward = 0
        for step in self.current_episode:
            total_reward += step["reward"]
        self.episodic_data["episode"].append(data["episode"])
        self.episodic_data["length"].append(len(self.current_episode))
        self.episodic_data["total_reward"].append(total_reward)
        self.episodic_data["avg_q"].append(tf.reduce_mean(self.driver_algorithm.get_values(self.random_states)).numpy())
        self.episodic_data["exploration"].append(data["exploration"])

    # def on_task_end(self, data=None):

    def save(self):
        tf.io.write_file(os.path.join(self.path, "random_states.tfw"), tf.io.serialize_tensor(self.random_states))
        pd.DataFrame(self.episodic_data).to_csv(os.path.join(self.path, "episodic_data.csv"), index=False)

    def load(self):
        try:
            self.episodic_data = pd.read_csv(os.path.join(self.path, "episodic_data.csv")).to_dict('list')
            self.random_states = tf.io.parse_tensor(tf.io.read_file(os.path.join(self.path, "random_states.tfw")),
                                                    tf.float32)
        except:
            print("No Random States found")
