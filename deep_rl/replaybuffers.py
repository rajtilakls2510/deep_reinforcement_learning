import tensorflow as tf
import os


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

    def __init__(self, max_transitions=1000, continuous=False):
        self.max_transitions = max_transitions
        # self.buffer = []
        self.current_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        if continuous:
            self.actions = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.next_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.terminals = tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False)
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
        buf_len = self.current_states.size()
        if buf_len <= batch_size:
            sampled_indices = tf.random.uniform(shape=(buf_len,), maxval=buf_len, dtype=tf.int32)
        else:
            sampled_indices = tf.random.uniform(shape=(batch_size,), maxval=buf_len, dtype=tf.int32)
        return self.current_states.gather(sampled_indices), self.actions.gather(sampled_indices), self.rewards.gather(
            sampled_indices), self.next_states.gather(sampled_indices), self.terminals.gather(sampled_indices)

    def save(self, path=""):
        tf.io.write_file(os.path.join(path, "current_states.tfw"), tf.io.serialize_tensor(self.current_states.stack()))
        tf.io.write_file(os.path.join(path, "actions.tfw"), tf.io.serialize_tensor(self.actions.stack()))
        tf.io.write_file(os.path.join(path, "rewards.tfw"), tf.io.serialize_tensor(self.rewards.stack()))
        tf.io.write_file(os.path.join(path, "next_states.tfw"), tf.io.serialize_tensor(self.next_states.stack()))
        tf.io.write_file(os.path.join(path, "terminals.tfw"), tf.io.serialize_tensor(self.terminals.stack()))

    def load(self, path=""):
        try:
            current_states = self.current_states.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "current_states.tfw")), tf.float32))
            actions = self.actions.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "actions.tfw")), tf.int32))
            rewards = self.rewards.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "rewards.tfw")), tf.float32))
            next_states = self.next_states.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "next_states.tfw")), tf.float32))
            terminals = self.terminals.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "terminals.tfw")), tf.bool))
            self.current_states = current_states
            self.actions = actions
            self.rewards = rewards
            self.next_states = next_states
            self.terminals = terminals
            self.current_index = self.current_states.size().numpy()
            print("Found", self.current_states.size().numpy(), "transitions")
        except:
            print("No Experience Replay found")
