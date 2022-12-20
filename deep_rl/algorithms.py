import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tqdm import tqdm
from deep_rl.replaybuffers import ExperienceReplay
from time import perf_counter


class DriverAlgorithm:
    # Base class for a training algorithm

    def __init__(self):
        self.env = None

    def set_env(self, env):
        self.env = env

    # Training code goes here
    def train(self, initial_episode, episodes, metric, batch_size=None):
        pass

    # Returns the next action to be taken, its value and whether it was a exploration step or not
    def get_action(self, state, explore=0.0):
        return tf.constant(0), tf.constant(0), tf.constant(
            False)  # Return: Action, Value for action, Action through exploration or not

    # Generated episodes and returns list frames for each episode
    def infer(self, episodes, metric, exploration=0.0):
        metric.on_task_begin()
        episode_data = []
        for _ in range(episodes):
            current_episode = []
            metric.on_episode_begin()
            self.env.reset()
            state, reward, frame = self.env.observe()
            state = tf.convert_to_tensor(state, tf.float32)
            while not self.env.is_episode_finished():
                action, action_, explored = self.get_action(state, exploration)
                self.env.take_action(action.numpy())
                state, reward, frame = self.env.observe()
                state = tf.convert_to_tensor(state, tf.float32)
                current_episode.append(
                    [frame, reward, state.numpy(), action.numpy(), action_.numpy(), explored.numpy()])
                metric.on_episode_step()
            episode_data.append(current_episode)
            metric.on_episode_end()
        metric.on_task_end()
        return episode_data

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

    def train(self, initial_episode, episodes, metric, batch_size=16):
        metric.load()
        metric.on_task_begin()
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):

            metric.on_episode_begin()

            self.env.reset()
            episode_start_chkpt = perf_counter()
            current_state, _, _ = self.env.observe()
            current_state = tf.convert_to_tensor(current_state, tf.float32)
            while not self.env.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action.numpy())
                next_state, reward, _ = self.env.observe()
                next_state = tf.convert_to_tensor(next_state, tf.float32)
                reward = tf.convert_to_tensor(reward, tf.float32)

                self.replay_buffer.insert_transition(
                    [current_state, action, reward, next_state,
                     tf.convert_to_tensor(self.env.is_episode_finished())])
                current_state = next_state
                metric.on_episode_step(
                    {
                        "action_value": action_value.numpy(),
                        "action": action.numpy(),
                        "reward": reward.numpy(),
                        "explored": explored.numpy()
                    }
                )
                update_start_chkpt = perf_counter()
                if self.step_counter % self.learn_after_steps == 0:
                    current_states, actions, rewards, next_states, terminals = self.replay_buffer.sample_batch_transitions(
                        batch_size=batch_size)
                    if current_states.shape[0] >= batch_size:
                        self._train_step(current_states, actions, rewards, next_states, terminals,
                                         current_states.shape[0])
                update_end_chkpt = perf_counter()
                self.step_counter += 1
                if self.step_counter % self.update_target_after == 0:
                    self.target_network.set_weights(self.q_network.get_weights())
                # print(round(update_end_chkpt - update_start_chkpt, 2))
            metric.on_episode_end({"episode": i, "exploration": self.exploration})
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

            metric.save()
            episode_end_chkpt = perf_counter()
            # print("Episode Completion: ", episode_end_chkpt - episode_start_chkpt)
        # Training End metric data storage

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


class DriverAlgorithmContinuous(DriverAlgorithm):

    # Generated episodes and returns list frames for each episode
    def infer(self, episodes, metric, exploration=0.0):
        metric.on_task_begin()
        episode_data = []
        for _ in range(episodes):
            current_episode = []
            metric.on_episode_begin()
            self.env.reset()
            state, reward, frame = self.env.observe()
            state = tf.convert_to_tensor(state, tf.float32)
            while not self.env.is_episode_finished():
                action = self.get_action(state, exploration)
                self.env.take_action(action.numpy())
                state, reward, frame = self.env.observe()
                state = tf.convert_to_tensor(state, tf.float32)
                values = self.get_values(tf.expand_dims(state, axis=0))[0]
                current_episode.append(
                    [frame, reward, state.numpy(), action.numpy(), values.numpy()])
                metric.on_episode_step()
            episode_data.append(current_episode)
            metric.on_episode_end()
        metric.on_task_end()
        return episode_data

    # Returns the action space
    def get_action(self, state, explore=0.0):
        return tf.constant(0)  # Return: Action

    # Return states after following a random policy
    def get_random_states(self, num_states=20):
        random_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self.env.reset()
        state, _, _ = self.env.observe()
        i = 0
        while i < num_states and not self.env.is_episode_finished():
            state = tf.convert_to_tensor(state, tf.float32)
            random_states = random_states.write(i, state)
            action = self.get_action(state)
            self.env.take_action(action.numpy())
            state, _, _ = self.env.observe()
            i += 1
        return random_states.stack()


class DeepDPG(DriverAlgorithmContinuous):

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

        # with tf.GradientTape(persistent=True) as actor_tape:
        #     actor_actions = self.actor_network(current_states)
        #     critic_actor_value = self.critic_network([current_states, actor_actions])
        #     dqda = actor_tape.gradient([critic_actor_value], [actor_actions])[0]
        #     target_a = dqda + actor_actions
        #     target_a = tf.stop_gradient(target_a)
        #     actor_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(target_a - actor_actions), axis=-1))

        with tf.GradientTape() as actor_tape:
            actor_loss = -tf.reduce_mean(self.critic_network([current_states, self.actor_network(current_states)]))

        actor_grads = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        self.actor_network.optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_weights))
        # del actor_tape

    @tf.function
    def update_targets(self, target_weights, weights, tau):
        for (target_w, w) in zip(target_weights, weights):
            target_w.assign(tau * w + (1 - tau) * target_w)

    def train(self, initial_episode, episodes, metric, batch_size=16):
        metric.load()
        metric.on_task_begin()
        for i in tqdm(range(initial_episode, initial_episode + episodes), desc="Episode"):

            metric.on_episode_begin()

            self.env.reset()
            episode_start_chkpt = perf_counter()
            current_state, _, _ = self.env.observe()
            current_state = tf.convert_to_tensor(current_state, tf.float32)
            while not self.env.is_episode_finished():
                action = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action.numpy())
                next_state, reward, _ = self.env.observe()
                next_state = tf.convert_to_tensor(next_state, tf.float32)
                reward = tf.convert_to_tensor(reward, tf.float32)

                self.replay_buffer.insert_transition(
                    [current_state, action, reward, next_state,
                     tf.convert_to_tensor(self.env.is_episode_finished())])
                current_state = next_state
                metric.on_episode_step(
                    {
                        "action": action.numpy(),
                        "reward": reward.numpy()
                    }
                )

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
            metric.on_episode_end({"episode": i, "exploration": self.exploration})
            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration /= self.exploration_decay
                if self.exploration < self.min_exploration:
                    self.exploration = self.min_exploration

            metric.save()
            episode_end_chkpt = perf_counter()
            # print("Episode Completion: ", episode_end_chkpt - episode_start_chkpt)
        # Training End metric data storage

    def get_action(self, state, explore=0.0):
        action = self.actor_network(tf.expand_dims(state, axis=0))[0]
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = action + tf.convert_to_tensor(self.env.get_random_action(), tf.float32)
        return action

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


class NeuralSarsa(DriverAlgorithm):

    def __init__(self, q_network: tf.keras.Model = None, learning_rate=0.01, discount_factor=0.9, exploration=0.0,
                 exploration_decay=1.1, min_exploration=0.0, exploration_decay_after=100):
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

            self.env.reset()
            current_state, _, _ = self.env.observe()
            while not self.env.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action)
                next_state, reward, _ = self.env.observe()
                next_action, next_value, _ = self.get_action(next_state, explore=self.exploration)
                current_state_tensor = tf.constant([current_state])
                with tf.GradientTape() as tape:
                    current_values = self.q_network(current_state_tensor)

                q_grads = tape.gradient(current_values, self.q_network.trainable_weights)

                if self.env.is_episode_finished():
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
            action = self.env.get_random_action()
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
                 exploration=0.0, exploration_decay=1.1, min_exploration=0.0, exploration_decay_after=100):
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

            self.env.reset()
            current_state, _, _ = self.env.observe()
            while not self.env.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action)
                next_state, reward, _ = self.env.observe()
                next_action, next_value, _ = self.get_action(next_state, explore=self.exploration)
                current_state_tensor = tf.constant([current_state])
                with tf.GradientTape() as tape:
                    current_values = self.q_network(current_state_tensor)

                q_grads = tape.gradient(current_values, self.q_network.trainable_weights)

                if self.env.is_episode_finished():
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
            action = self.env.get_random_action()
            explored = True
        return action, action_[action].numpy(), explored  # Action, Value for Action, explored or not

    def get_values(self, states):
        return tf.reduce_max(self.q_network(tf.constant(states)), axis=1)

    def save(self, path=""):
        self.q_network.save(os.path.join(path, "q_network"))

    def load(self, path=""):
        self.q_network = load_model(os.path.join(path, "q_network"))
