import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tqdm import tqdm
from deep_rl.replaybuffers import ExperienceReplay


class DriverAlgorithm:
    # Base class for a training algorithm

    def __init__(self):
        self.interpreter = None

    def set_interpreter(self, interpreter):
        self.interpreter = interpreter

    # Training code goes here
    def train(self, initial_episode, episodes, metric, batch_size=None):
        pass

    # Returns the next action to be takes, its value and whether it was a exploration step or not
    def get_action(self, state, explore=0.0):
        return 0, 0, False  # Return: Action, Value for action, Action through exploration or not

    # Generated episodes and returns list frames for each episode
    def infer(self, episodes, metric, exploration=0.0):
        metric.on_task_begin()
        rgb_array = []
        for _ in range(episodes):
            current_episode = []
            metric.on_episode_begin()
            self.interpreter.reset()
            state, reward, frame = self.interpreter.observe()
            while not self.interpreter.is_episode_finished():
                current_episode.append(frame)
                action, action_, explored = self.get_action(state, exploration)
                self.interpreter.take_action(action)
                state, reward, frame = self.interpreter.observe()
                metric.on_episode_step()
            rgb_array.append(current_episode)
            metric.on_episode_end()
        metric.on_task_end()
        return np.array(rgb_array)

    # Return states after following a random policy
    def get_random_states(self, num_states=20):
        random_states = []
        self.interpreter.reset()
        state, _, _ = self.interpreter.observe()
        while num_states > 0 and not self.interpreter.is_episode_finished():
            random_states.append(state)
            action, action_, _ = self.get_action(state)
            self.interpreter.take_action(action)
            state, _, _ = self.interpreter.observe()
            num_states -= 1
        return random_states

    # Return Q values for a list of states
    def get_values(self, states):
        pass

    # Saves restorable parameters like networks and metrics
    def save(self, path=""):
        pass

    # Loads restorable parameters like networks and metrics
    def load(self, path=""):
        pass


class DeepQLearning(DriverAlgorithm):

    def __init__(self, q_network: tf.keras.Model = None, optimizer: tf.keras.optimizers.Optimizer = None,
                 replay_size=1000,
                 discount_factor=0.9, exploration=0.1, min_exploration=0.1, exploration_decay=1.1,
                 exploration_decay_after=100,
                 update_target_after_steps=100):
        super().__init__()
        self.q_network = q_network
        if self.q_network is None:
            self.target_network = None
        else:
            self.target_network = clone_model(self.q_network)
        self.optimizer = optimizer
        self.replay_buffer = ExperienceReplay(replay_size)
        self.exploration = exploration
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.discount_factor = discount_factor
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.update_target_after = update_target_after_steps
        self.step_counter = 0

    def train(self, initial_episode, episodes, metric, batch_size=16):
        metric.load()
        metric.on_task_begin()
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):

            metric.on_episode_begin()

            self.interpreter.reset()
            current_state, _, _ = self.interpreter.observe()
            while not self.interpreter.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.interpreter.take_action(action)
                next_state, reward, _ = self.interpreter.observe()

                self.replay_buffer.insert_transition(
                    [current_state, action, reward, next_state, self.interpreter.is_episode_finished()])

                sampled_transitions = self.replay_buffer.sample_batch_transitions(batch_size=batch_size)

                next_action_values = tf.reduce_max(self.target_network(sampled_transitions[3]), axis=1)
                targets = []
                for r, nav, term in zip(sampled_transitions[2], next_action_values, sampled_transitions[4]):
                    targets.append([(r + self.discount_factor * nav).numpy() if not term else r.numpy()])
                targets = tf.constant(targets)
                with tf.GradientTape() as tape:
                    preds = self.q_network(sampled_transitions[0])
                    batch_nums = tf.range(0, limit=preds.get_shape().as_list()[0])
                    indices = tf.stack((batch_nums, sampled_transitions[1]), axis=1)
                    new_preds = tf.gather_nd(preds, indices)
                    loss = self.loss_fn(targets, new_preds)

                grads = tape.gradient(loss, self.q_network.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))

                current_state = next_state

                metric.on_episode_step(
                    {
                        "action_value": action_value,
                        "action": action,
                        "reward": reward,
                        "explored": explored
                    }
                )
                self.step_counter += 1
                if self.step_counter % self.update_target_after == 0:
                    self.target_network.set_weights(self.q_network.get_weights())

            metric.on_episode_end({"episode": i, "exploration": self.exploration})
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

            metric.save()
        # Training End metric data storage

    def get_action(self, state, explore=0.0):
        action_ = self.q_network(tf.constant([state]))[0]
        action = tf.argmax(action_).numpy()
        explored = False
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = self.interpreter.get_randomized_action()
            explored = True
        return action, action_[action].numpy(), explored  # Action, Value for Action, explored or not

    def get_values(self, states):
        return tf.reduce_max(self.q_network(tf.constant(states)), axis=1)

    def save(self, path=""):
        self.q_network.save(os.path.join(path, "q_network"))
        self.target_network.save(os.path.join(path, "target_network"))

    def load(self, path=""):
        self.q_network = load_model(os.path.join(path, "q_network"))
        try:
            self.target_network = load_model(os.path.join(path, "target_network"))
        except:
            if self.q_network is not None:
                self.target_network = clone_model(self.q_network)


class NeuralSarsa(DriverAlgorithm):

    def __init__(self, q_network: tf.keras.Model = None, learning_rate=0.01, discount_factor=0.9, exploration=0.0,
                 exploration_decay=1.1, min_exploration=0.1, exploration_decay_after=100):
        super(NeuralSarsa, self).__init__()
        self.q_network = q_network
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.min_exploration = min_exploration

    def train(self, initial_episode, episodes, metric, batch_size=None):
        metric.load()
        metric.on_task_begin()
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):

            metric.on_episode_begin()

            self.interpreter.reset()
            current_state, _, _ = self.interpreter.observe()
            while not self.interpreter.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.interpreter.take_action(action)
                next_state, reward, _ = self.interpreter.observe()
                next_action, next_value, _ = self.get_action(next_state, explore=self.exploration)
                current_state_tensor = tf.constant([current_state])
                with tf.GradientTape() as tape:
                    current_values = self.q_network(current_state_tensor)

                q_grads = tape.gradient(current_values, self.q_network.trainable_weights)

                if self.interpreter.is_episode_finished():
                    delta = reward - current_values[0][action]
                else:
                    delta = reward + self.discount_factor * next_value - current_values[0][action]

                for j in range(len(q_grads)):
                    self.q_network.trainable_weights[j].assign_add(self.learning_rate * delta * q_grads[j])

                current_state = next_state

                metric.on_episode_step(
                    {
                        "action_value": action_value,
                        "action": action,
                        "reward": reward,
                        "explored": explored
                    }
                )

            metric.on_episode_end({"episode": i, "exploration": self.exploration})
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

            metric.save()
        # Training End metric data storage

    def get_action(self, state, explore=0.0):
        action_ = self.q_network(tf.constant([state]))[0]
        action = tf.argmax(action_).numpy()
        explored = False
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = self.interpreter.get_randomized_action()
            explored = True
        return action, action_[action].numpy(), explored  # Action, Value for Action, explored or not

    def get_values(self, states):
        return tf.reduce_max(self.q_network(tf.constant(states)), axis=1)

    def save(self, path=""):
        self.q_network.save(os.path.join(path, "q_network"))

    def load(self, path=""):
        self.q_network = load_model(os.path.join(path, "q_network"))


class NeuralSarsaLambda(DriverAlgorithm):

    def __init__(self, q_network: tf.keras.Model = None, learning_rate=0.01, discount_factor=0.9, lmbda=0.9,
                 exploration=0.0, exploration_decay=1.1, min_exploration=0.1, exploration_decay_after=100):
        super(NeuralSarsaLambda, self).__init__()
        self.q_network = q_network
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lmbda = lmbda
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.min_exploration = min_exploration

    def train(self, initial_episode, episodes, metric, batch_size=None):
        metric.load()
        metric.on_task_begin()
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):

            metric.on_episode_begin()

            el_trace = []
            for j in range(len(self.q_network.trainable_weights)):
                el_trace.append(tf.zeros(self.q_network.trainable_weights[j].shape))

            self.interpreter.reset()
            current_state, _, _ = self.interpreter.observe()
            while not self.interpreter.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.interpreter.take_action(action)
                next_state, reward, _ = self.interpreter.observe()
                next_action, next_value, _ = self.get_action(next_state, explore=self.exploration)
                current_state_tensor = tf.constant([current_state])
                with tf.GradientTape() as tape:
                    current_values = self.q_network(current_state_tensor)

                q_grads = tape.gradient(current_values, self.q_network.trainable_weights)

                if self.interpreter.is_episode_finished():
                    delta = reward - current_values[0][action]
                else:
                    delta = reward + self.discount_factor * next_value - current_values[0][action]

                for j in range(len(q_grads)):
                    el_trace[j] = self.lmbda * self.discount_factor * el_trace[j] + q_grads[j]
                for j in range(len(el_trace)):
                    self.q_network.trainable_weights[j].assign_add(self.learning_rate * delta * el_trace[j])

                current_state = next_state

                metric.on_episode_step(
                    {
                        "action_value": action_value,
                        "action": action,
                        "reward": reward,
                        "explored": explored
                    }
                )

            metric.on_episode_end({"episode": i, "exploration": self.exploration})
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

            metric.save()
        # Training End metric data storage

    def get_action(self, state, explore=0.0):
        action_ = self.q_network(tf.constant([state]))[0]
        action = tf.argmax(action_).numpy()
        explored = False
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = self.interpreter.get_randomized_action()
            explored = True
        return action, action_[action].numpy(), explored  # Action, Value for Action, explored or not

    def get_values(self, states):
        return tf.reduce_max(self.q_network(tf.constant(states)), axis=1)

    def save(self, path=""):
        self.q_network.save(os.path.join(path, "q_network"))

    def load(self, path=""):
        self.q_network = load_model(os.path.join(path, "q_network"))
