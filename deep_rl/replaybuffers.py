import tensorflow as tf
import os, pickle


class ReplayBuffer:

    # Inserts transition to buffer
    def insert_transition(self, transition):
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
        # self.buffer = []
        self.current_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.next_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.terminals = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        self.current_index = 0

    def _insert_transition_at(self, transition, index):
        self.current_states = self.current_states.write(index, transition[0])
        self.actions = self.actions.write(index, transition[1])
        self.rewards = self.rewards.write(index, transition[2])
        self.next_states = self.next_states.write(index, transition[3])
        self.terminals = self.terminals.write(index, transition[4])

    # Transition Format: current_state, action, reward, next_state, terminal step or not
    def insert_transition(self, transition):
        self._insert_transition_at(transition, self.current_index % self.max_transitions)
        self.current_index += 1

    def sample_batch_transitions(self, batch_size=16):
        buf_len = self.current_states.size().numpy()
        if buf_len <= batch_size:
            sampled_indices = tf.random.uniform(shape=(buf_len,), maxval=buf_len, dtype=tf.int32)
        else:
            sampled_indices = tf.random.uniform(shape=(batch_size,), maxval=batch_size, dtype=tf.int32)
        # sampled_current_states = []
        # sampled_actions = []
        # sampled_rewards = []
        # sampled_next_states = []
        # sampled_terminal_step = []
        # for index in sampled_indices:
        #     sampled_current_states.append(self.buffer[index][0])
        #     sampled_actions.append(self.buffer[index][1])
        #     sampled_rewards.append(self.buffer[index][2])
        #     sampled_next_states.append(self.buffer[index][3])
        #     sampled_terminal_step.append(self.buffer[index][4])
        # return tf.constant(sampled_current_states), tf.constant(sampled_actions), tf.constant(
        #     sampled_rewards, dtype=tf.float32), tf.constant(sampled_next_states), tf.constant(sampled_terminal_step)
        return self.current_states.gather(sampled_indices), self.actions.gather(sampled_indices), self.rewards.gather(
            sampled_indices), self.next_states.gather(sampled_indices), self.terminals.gather(sampled_indices)

    def save(self, path=""):
        pass
        # try:
        #     pickle.dump(self.buffer, open(os.path.join(path, "experience_replay.pkl"), "wb"))
        # except:
        #     os.makedirs(path)
        #     pickle.dump(self.buffer, open(os.path.join(path, "experience_replay.pkl"), "wb"))

    def load(self, path=""):
        pass
        # try:
        #     self.buffer = pickle.load(open(os.path.join(path, "experience_replay.pkl"), "rb"))
        #     print("Found " + str(len(self.buffer)) + " transitions")
        # except:
        #     print("No Experience Replay found")
