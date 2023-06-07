import os
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tqdm import tqdm
from deep_rl.replaybuffers import ExperienceReplay
from abc import ABC, abstractmethod
from deep_rl.analytics import Metric


class DriverAlgorithm(ABC):
    # Base abstract class for a driver algorithm
    # Description: This class is the base class for a driver algorithm which drives an agent.
    #           It contains methods which can run training steps and evaluation steps. By default,
    #           an implementation of evaluation run is given by through infer() method.

    def __init__(self):
        self.env = None
        self.step_counter = 1

    def set_env(self, env):
        self.env = env

    # Training code goes here
    @abstractmethod
    def train(self, initial_episode, episodes, metric, batch_size=None):
        pass

    # Returns the next action to be taken, its value and whether it was an exploration step or not
    @abstractmethod
    def get_action(self, state, explore=0.0):
        return tf.constant(0), tf.constant(0), tf.constant(
            False)  # Return: Action, Value for action, Action through exploration or not

    # Generated episodes and returns list frames for each episode
    def infer(self,initial_episode, episodes, metrics: list[Metric] = (), exploration=0.0):
        for metric in metrics: metric.on_task_begin()

        # Evaluating for multiple episodes
        for ep in range(initial_episode, initial_episode+episodes):
            episode_data = {"episode": ep, "exploration": exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_begin(episode_data)

            # Evaluation loop for an episode
            self.env.reset()
            current_state, reward, frame = self.env.observe()
            current_state = tf.convert_to_tensor(current_state, tf.float32)
            while not self.env.is_episode_finished():
                # Interaction Step
                action, action_value, explored = self.get_action(current_state, exploration)
                self.env.take_action(action.numpy())
                next_state, reward, frame = self.env.observe()
                next_state = tf.convert_to_tensor(next_state, tf.float32)

                # Sending step data to metrics
                step_data = {
                    "current_state": current_state.numpy(),
                    "action_value": action_value.numpy(),
                    "action": action.numpy(),
                    "reward": reward,
                    "next_state": next_state.numpy(),
                    "explored": explored.numpy(),
                    "frame": frame,
                }
                for metric in metrics: metric.on_episode_step(step_data)
                # self.step_counter += 1
                current_state = next_state
            episode_data = {"episode": ep, "exploration": exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_end(episode_data)
        for metric in metrics: metric.on_task_end()

    # Return states after following a random policy
    def get_random_states(self, num_states=20):
        random_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.env.reset()
        state, _, _ = self.env.observe()
        i = 0
        while i < num_states and not self.env.is_episode_finished():
            state = tf.convert_to_tensor(state, tf.float32)
            random_states = random_states.write(i, state)
            action, action_, _ = self.get_action(state)
            self.env.take_action(action.numpy())
            state, _, _ = self.env.observe()
            i += 1
        return random_states.stack()

    # Return Q values for a list of states
    def get_values(self, states):
        return tf.constant(0)

    # Saves restorable parameters like networks and metrics
    def save(self, path=""):
        pass

    # Loads restorable parameters like networks and metrics
    def load(self, path=""):
        pass


class DeepQLearning(DriverAlgorithm):
    # This is an Implementation of the DQN Algorithm

    def __init__(self, q_network: tf.keras.Model = None,
                 loss=tf.keras.losses.Huber(),
                 learn_after_steps=1,
                 replay_size=1000,
                 discount_factor=0.9, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
                 exploration_decay_after=100,
                 update_target_after_steps=100):
        super().__init__()
        self.q_network = q_network
        if self.q_network is None:
            self.target_network = None
        else:
            self.target_network = clone_model(self.q_network)
        self.replay_buffer = ExperienceReplay(replay_size)
        self.loss_fn = loss
        self.learn_after_steps = learn_after_steps
        self.exploration = exploration
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.discount_factor = tf.convert_to_tensor(discount_factor)
        self.update_target_after = update_target_after_steps
        self.step_counter = 1

    @tf.function
    def _train_step(self, current_states, actions, rewards, next_states, terminals, batch_size):
        next_action_values = tf.reduce_max(self.target_network(next_states), axis=1)
        targets = tf.where(terminals, rewards, rewards + self.discount_factor * next_action_values)

        with tf.GradientTape() as tape:
            preds = self.q_network(current_states)
            batch_nums = tf.range(0, limit=batch_size)
            indices = tf.stack((batch_nums, actions), axis=1)
            new_preds = tf.gather_nd(preds, indices)
            loss = self.loss_fn(targets, new_preds)
        grads = tape.gradient(loss, self.q_network.trainable_weights)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))

    def train(self, initial_episode, episodes, metrics: list[Metric], batch_size=16):
        for metric in metrics: metric.on_task_begin()

        # Training for multiple episodes
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):
            episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_begin(episode_data)

            # Training loop for an episode
            self.env.reset()
            current_state, _, _ = self.env.observe()
            current_state = tf.convert_to_tensor(current_state, tf.float32)
            while not self.env.is_episode_finished():
                # Interaction step
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action.numpy())
                next_state, reward, frame = self.env.observe()
                next_state = tf.convert_to_tensor(next_state, tf.float32)
                reward = tf.convert_to_tensor(reward, tf.float32)

                # Inserting transition to replay buffer
                self.replay_buffer.insert_transition(
                    [current_state, action, reward, next_state,
                     tf.convert_to_tensor(self.env.is_episode_finished())])
                current_state = next_state

                # Sending step data to metrics
                step_data = {
                    "current_state": current_state.numpy(),
                    "action_value": action_value.numpy(),
                    "action": action.numpy(),
                    "reward": reward.numpy(),
                    "next_state": next_state.numpy(),
                    "explored": explored.numpy(),
                    "frame": frame
                }
                for metric in metrics: metric.on_episode_step(step_data)
                # Learning from a batch of transitions
                if self.step_counter % self.learn_after_steps == 0:
                    current_states, actions, rewards, next_states, terminals = self.replay_buffer.sample_batch_transitions(
                        batch_size=batch_size)
                    if current_states.shape[0] >= batch_size:
                        self._train_step(current_states, actions, rewards, next_states, terminals,
                                         current_states.shape[0])

                self.step_counter += 1

                # Updating Target Network Parameters
                if self.step_counter % self.update_target_after == 0:
                    self.target_network.set_weights(self.q_network.get_weights())

            episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_end(episode_data)

            # Decaying Exploration Parameter
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

        for metric in metrics: metric.on_task_end()

    def get_action(self, state, explore=0.0):
        action_ = self.q_network(tf.expand_dims(state, axis=0))[0]
        action = tf.argmax(action_, output_type=tf.int32)
        explored = tf.constant(False)
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = tf.convert_to_tensor(self.env.get_random_action(), tf.int32)
            explored = tf.constant(True)
        return action, action_[action], explored  # Action, Value for Action, explored or not

    def get_values(self, states):
        return tf.reduce_max(self.q_network(states), axis=1)

    def save(self, path=""):
        self.q_network.save(os.path.join(path, "q_network"))
        self.target_network.save(os.path.join(path, "target_network"))
        self.replay_buffer.save(os.path.join(path, "replay"))

    def load(self, path=""):
        self.q_network = load_model(os.path.join(path, "q_network"))
        try:
            self.target_network = load_model(os.path.join(path, "target_network"))
        except:
            if self.q_network is not None:
                self.target_network = clone_model(self.q_network)
        self.replay_buffer.load(os.path.join(path, "replay"))


class DoubleDeepQLearning(DeepQLearning):
    # This class is an implementation of the DoubleDQN Algorithm

    def __init__(self, q_network: tf.keras.Model = None,
                 loss=tf.keras.losses.Huber(),
                 learn_after_steps=1,
                 replay_size=1000,
                 discount_factor=0.9, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
                 exploration_decay_after=100,
                 update_target_after_steps=100):
        super().__init__(q_network, loss, learn_after_steps, replay_size, discount_factor, exploration, min_exploration,
                         exploration_decay, exploration_decay_after, update_target_after_steps)

    @tf.function
    def _train_step(self, current_states, actions, rewards, next_states, terminals, batch_size):
        next_actions = tf.argmax(self.q_network(next_states), axis=1, output_type=tf.int32)
        next_action_values = self.target_network(next_states)
        batch_nums = tf.range(0, limit=batch_size)
        indices = tf.stack((batch_nums, next_actions), axis=1)
        next_action_values = tf.gather_nd(next_action_values, indices)
        targets = tf.where(terminals, rewards, rewards + self.discount_factor * next_action_values)

        with tf.GradientTape() as tape:
            preds = self.q_network(current_states)
            batch_nums = tf.range(0, limit=batch_size)
            indices = tf.stack((batch_nums, actions), axis=1)
            new_preds = tf.gather_nd(preds, indices)
            loss = self.loss_fn(targets, new_preds)
        grads = tape.gradient(loss, self.q_network.trainable_weights)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))


class DeepDPG(DriverAlgorithm):
    # This class is an implementation of the DeepDPG Algorithm

    def __init__(self, actor_network: tf.keras.Model = None, critic_network: tf.keras.Model = None, learn_after_steps=1,
                 replay_size=1000, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
                 exploration_decay_after=100, discount_factor=0.9, tau=0.001):
        super().__init__()
        self.actor_network = actor_network
        self.critic_network = critic_network
        if self.actor_network is None:
            self.actor_target_network = None
        else:
            self.actor_target_network = clone_model(self.actor_network)

        if self.critic_network is None:
            self.critic_target_network = None
        else:
            self.critic_target_network = clone_model(self.critic_network)

        self.learn_after_steps = learn_after_steps
        self.replay_buffer = ExperienceReplay(replay_size, continuous=True)
        self.discount_factor = tf.convert_to_tensor(discount_factor)
        self.exploration = exploration
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.tau = tf.convert_to_tensor(tau)
        self.step_counter = 1
        self.critic_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def _train_step(self, current_states, actions, rewards, next_states):

        targets = tf.expand_dims(rewards, axis=1) + self.discount_factor * self.critic_target_network(
            [next_states, self.actor_target_network(next_states)])

        with tf.GradientTape() as critic_tape:
            critic_value = self.critic_network([current_states, actions])
            critic_loss = self.critic_loss(targets, critic_value)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_network.trainable_weights)
        self.critic_network.optimizer.apply_gradients(zip(critic_grads, self.critic_network.trainable_weights))

        with tf.GradientTape() as actor_tape:
            actor_loss = -tf.reduce_mean(self.critic_network([current_states, self.actor_network(current_states)]))

        actor_grads = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        self.actor_network.optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_weights))

    @tf.function
    def update_targets(self, target_weights, weights, tau):
        for (target_w, w) in zip(target_weights, weights):
            target_w.assign(tau * w + (1 - tau) * target_w)

    def train(self, initial_episode, episodes, metrics, batch_size=16):
        for metric in metrics: metric.on_task_begin()

        # Training for multiple episodes
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):
            episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_begin(episode_data)

            # Training loop for an episode
            self.env.reset()
            current_state, _, _ = self.env.observe()
            current_state = tf.convert_to_tensor(current_state, tf.float32)
            while not self.env.is_episode_finished():
                # Interaction Step
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action.numpy())
                next_state, reward, frame = self.env.observe()
                next_state = tf.convert_to_tensor(next_state, tf.float32)
                reward = tf.convert_to_tensor(reward, tf.float32)

                # Inserting transition in replay buffer
                self.replay_buffer.insert_transition(
                    [current_state, action, reward, next_state,
                     tf.convert_to_tensor(self.env.is_episode_finished())])
                current_state = next_state

                # Sending step data to metrics
                step_data = {
                        "current_state": current_state.numpy(),
                        "action_value": action_value.numpy(),
                        "action": action.numpy(),
                        "reward": reward.numpy(),
                        "next_state": next_state.numpy(),
                        "explored": explored.numpy(),
                        "frame": frame
                    }
                for metric in metrics: metric.on_episode_step(step_data)

                # Learning from a batch of transitions
                if self.step_counter % self.learn_after_steps == 0:
                    current_states, actions, rewards, next_states, _ = self.replay_buffer.sample_batch_transitions(
                        batch_size=batch_size)

                    if current_states.shape[0] >= batch_size:
                        self._train_step(current_states, actions, rewards, next_states)
                        self.update_targets(self.actor_target_network.trainable_weights,
                                            self.actor_network.trainable_weights, self.tau)
                        self.update_targets(self.critic_target_network.trainable_weights,
                                            self.critic_network.trainable_weights, self.tau)

                self.step_counter += 1
            episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_end(episode_data)

            # Decaying Exploration Parameter
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

        for metric in metrics: metric.on_task_end()

    def get_action(self, state, explore=0.0):
        state = tf.expand_dims(state, axis=0)
        action = self.actor_network(state)
        explored = tf.constant(False)
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = action + tf.convert_to_tensor(self.env.get_random_action(), tf.float32)
            explored = tf.constant(True)
        value = self.critic_network([state, action])
        return action[0], value[0][0], explored

    def get_values(self, states):
        return self.critic_network([states, self.actor_network(states)])

    def save(self, path=""):
        self.actor_network.save(os.path.join(path, "actor_network"))
        self.actor_target_network.save(os.path.join(path, "actor_target_network"))
        self.critic_network.save(os.path.join(path, "critic_network"))
        self.critic_target_network.save(os.path.join(path, "critic_target_network"))
        self.replay_buffer.save(os.path.join(path, "replay"))

    def load(self, path=""):
        self.actor_network = load_model(os.path.join(path, "actor_network"))
        self.critic_network = load_model(os.path.join(path, "critic_network"))
        try:
            self.actor_target_network = load_model(os.path.join(path, "actor_target_network"))
        except:
            if self.actor_network is not None:
                self.actor_target_network = clone_model(self.actor_network)
        try:
            self.critic_target_network = load_model(os.path.join(path, "critic_target_network"))
        except:
            if self.critic_network is not None:
                self.critic_target_network = clone_model(self.critic_network)
        self.replay_buffer.load(os.path.join(path, "replay"))

