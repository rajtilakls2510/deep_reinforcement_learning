import tensorflow as tf
import os, pickle


class ReplayBuffer:

    # Inserts transition to buffer
    def insert_transition(self, transition):
        pass

    # Samples and returns a transition from buffer
    def sample_transition(self):
        pass

    def sample_batch_transitions(self, batch_size=16):
        pass

    def save(self, path=""):
        pass

    def load(self, path=""):
        pass


class ExperienceReplay(ReplayBuffer):

    def __init__(self, max_transitions=1000):
        self.max_transitions = max_transitions
        self.buffer = []

    # Transition Format: current_state, action, reward, next_state, terminal step or not
    def insert_transition(self, transition):
        if len(self.buffer) >= self.max_transitions:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample_transition(self):
        return self.buffer[tf.random.uniform(shape=(1,), maxval=len(self.buffer), dtype=tf.int32)]

    def sample_batch_transitions(self, batch_size=16):
        buf_len = len(self.buffer)
        if buf_len<=batch_size:
            sampled_indices = tf.random.uniform(shape=(buf_len,), maxval=buf_len, dtype=tf.int32)
        else:
            sampled_indices = tf.random.uniform(shape=(batch_size,), maxval=len(self.buffer), dtype=tf.int32)
        sampled_current_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_terminal_step = []
        for index in sampled_indices:
            sampled_current_states.append(self.buffer[index][0])
            sampled_actions.append(self.buffer[index][1])
            sampled_rewards.append(self.buffer[index][2])
            sampled_next_states.append(self.buffer[index][3])
            sampled_terminal_step.append(self.buffer[index][4])
        return tf.constant(sampled_current_states), tf.constant(sampled_actions), tf.constant(
            sampled_rewards, dtype=tf.float32), tf.constant(sampled_next_states), tf.constant(sampled_terminal_step)

    def save(self, path=""):
        try:
            pickle.dump(self.buffer, open(os.path.join(path, "experience_replay.pkl"), "wb"))
        except:
            os.makedirs(path)
            pickle.dump(self.buffer, open(os.path.join(path, "experience_replay.pkl"), "wb"))

    def load(self, path=""):
        try:
            self.buffer = pickle.load(open(os.path.join(path, "experience_replay.pkl"), "rb"))
            print("Found "+str(len(self.buffer))+" transitions")
        except:
            print("No Experience Replay found")
