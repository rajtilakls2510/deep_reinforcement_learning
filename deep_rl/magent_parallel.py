from deep_rl.agent import DRLEnvironment
from deep_rl.algorithms import DriverAlgorithm
from deep_rl.analytics import Metric
from deep_rl.replaybuffers import ReplayBuffer
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from deep_rl.utils import set_weights, get_weights
import os
import pandas as pd
import numpy as np


class MAParallelEnvironment(DRLEnvironment):

    def __init__(self, env):
        self.env = env
        self.observations, _ = self.env.reset()
        self.agents = [agent for agent in self.env.possible_agents if "agent" in agent]
        self.adversaries = [agent for agent in self.env.possible_agents if "adversary" in agent]
        self.terminated = {}
        self.truncated = {}
        self.rewards = {}
        self.preprocess_observations = {}

    def observe(self):
        # {agent1: observation, agent2: obs....}, {agent1: reward, ...}, {agent1: terminated...}, frame
        frame = self.env.render()
        self.preprocess_observations = self.preprocess_state(self.observations)
        self.rewards = self.calculate_reward()
        return self.preprocess_observations, self.rewards, self.terminated, frame

    def take_action(self, actions):
        # actions: {agent1: action, agent2: action ...}
        self.observations, self.rewards, self.terminated, self.truncated, _ = self.env.step({agent: tf.clip_by_value(actions[agent], 0, 1) for agent in self.agents+self.adversaries})
        self.terminated = {agent: self.terminated[agent] or self.truncated[agent] for agent in self.agents+self.adversaries}

    def calculate_reward(self, **kwargs):
        return self.rewards

    def get_random_action(self, agent):
        return tf.random.normal(shape=(5,), mean=0.5, stddev=0.1)

    def is_episode_finished(self):
        return len(self.env.agents) == 0

    def close(self):
        self.env.close()

    def reset(self):
        self.observations, _ = self.env.reset()


# class MADDPG(DriverAlgorithm):
#
#     def __init__(self, env: MAParallelEnvironment, agents, adversaries, actor_network=None, actor_adversary_network=None, critic_network=None, critic_adversary_network=None, learn_after_steps=1,
#                  replay_size=1000, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
#                  exploration_decay_after=100, discount_factor=0.9, tau=0.001):
#         super().__init__()
#         self.env = env
#         self.agents = agents
#         self.adversaries = adversaries
#
#         if actor_network and critic_network:
#             self.actor_networks = {agent: clone_model(actor_network) for agent in self.agents}
#             self.actor_networks.update({agent: clone_model(actor_adversary_network) for agent in self.adversaries})
#             for an in self.actor_networks.values():
#                 an.compile(optimizer=Adam.from_config(actor_network.optimizer.get_config()))
#             self.target_actor_networks = {agent: clone_model(actor_network) for agent in self.agents}
#             self.target_actor_networks.update({agent: clone_model(actor_adversary_network) for agent in self.adversaries})
#             self.critic_networks = {agent: clone_model(critic_network) for agent in self.agents}
#             self.critic_networks.update({agent: clone_model(critic_adversary_network) for agent in self.adversaries})
#             for cn in self.critic_networks.values():
#                 cn.compile(optimizer=Adam.from_config(critic_network.optimizer.get_config()))
#             self.target_critic_networks = {agent: clone_model(critic_network) for agent in self.agents}
#             self.target_critic_networks.update({agent: clone_model(critic_adversary_network) for agent in self.adversaries})
#             self.current_actor_network, self.current_critic_network = clone_model(actor_network), clone_model(critic_network)
#             self.current_target_actor_network, self.current_target_critic_network = clone_model(actor_network), clone_model(critic_network)
#             self.current_actor_adversary_network, self.current_critic_adversary_network = clone_model(actor_adversary_network), clone_model(
#                 critic_adversary_network)
#             self.current_target_actor_adversary_network, self.current_target_critic_adversary_network = clone_model(
#                 actor_adversary_network), clone_model(critic_adversary_network)
#         else:
#             self.actor_networks = None
#             self.target_actor_networks = None
#             self.critic_networks = None
#             self.target_critic_networks = None
#             self.current_actor_network, self.current_critic_network = None, None
#             self.current_target_actor_network, self.current_target_critic_network = None, None
#             self.current_actor_adversary_network, self.current_critic_adversary_network = None, None
#             self.current_target_actor_adversary_network, self.current_target_critic_adversary_network = None, None
#
#         self.learn_after_steps = learn_after_steps
#         self.replay_buffer = MAExperienceReplay(agents=self.agents+self.adversaries, max_transitions=replay_size, continuous=True)
#         self.discount_factor = tf.convert_to_tensor(discount_factor)
#         self.exploration = exploration
#         self.min_exploration = min_exploration
#         self.exploration_decay = exploration_decay
#         self.exploration_decay_after = exploration_decay_after
#         self.tau = tf.convert_to_tensor(tau)
#         self.step_counter = 1
#         self.critic_loss = tf.keras.losses.MeanSquaredError()
#
#     @tf.function
#     def _critic_agent_train_step(self, current_observations, actions, rewards, next_observations, next_actions):
#         targets = tf.expand_dims(rewards, axis=1) + self.discount_factor * self.current_target_critic_network([next_observations, next_actions])
#
#         with tf.GradientTape() as tape:
#             critic_value = self.current_critic_network([current_observations, actions])
#             critic_loss = self.critic_loss(targets, critic_value)
#
#         critic_grads = tape.gradient(critic_loss, self.current_critic_network.trainable_variables)
#         self.current_critic_network.optimizer.apply_gradients(zip(critic_grads, self.current_critic_network.trainable_variables))
#
#     @tf.function
#     def _critic_adversary_train_step(self, current_observations, actions, rewards, next_observations, next_actions):
#         targets = tf.expand_dims(rewards, axis=1) + self.discount_factor * self.current_target_critic_adversary_network(
#             [next_observations, next_actions])
#
#         with tf.GradientTape() as tape:
#             critic_value = self.current_critic_adversary_network([current_observations, actions])
#             critic_loss = self.critic_loss(targets, critic_value)
#
#         critic_grads = tape.gradient(critic_loss, self.current_critic_adversary_network.trainable_variables)
#         self.current_critic_adversary_network.optimizer.apply_gradients(
#             zip(critic_grads, self.current_critic_adversary_network.trainable_variables))
#
#     @tf.function
#     def _actor_agent_train_step(self, current_observations, actions, current_observation, agent):
#         indices = tf.expand_dims(tf.range(current_observations.shape[0]), axis=1)
#         filler = tf.fill(dims=(current_observations.shape[0], 1), value=agent)
#         indices = tf.concat([indices, filler], axis=1)
#         with tf.GradientTape() as tape:
#             new_actions = tf.tensor_scatter_nd_update(actions, indices, self.current_actor_network(current_observation))
#             actor_loss = -tf.reduce_mean(self.current_critic_network([current_observations, new_actions]))
#         actor_grads = tape.gradient(actor_loss, self.current_actor_network.trainable_variables)
#         self.current_actor_network.optimizer.apply_gradients(zip(actor_grads, self.current_actor_network.trainable_variables))
#
#     @tf.function
#     def _actor_adversary_train_step(self, current_observations, actions, current_observation, agent):
#         indices = tf.expand_dims(tf.range(current_observations.shape[0]), axis=1)
#         filler = tf.fill(dims=(current_observations.shape[0], 1), value=agent)
#         indices = tf.concat([indices, filler], axis=1)
#         with tf.GradientTape() as tape:
#             new_actions = tf.tensor_scatter_nd_update(actions, indices,
#                                                       self.current_actor_adversary_network(current_observation))
#             actor_loss = -tf.reduce_mean(self.current_critic_adversary_network([current_observations, new_actions]))
#         actor_grads = tape.gradient(actor_loss, self.current_actor_adversary_network.trainable_variables)
#         self.current_actor_adversary_network.optimizer.apply_gradients(
#             zip(actor_grads, self.current_actor_adversary_network.trainable_variables))
#
#     @tf.function
#     def update_targets(self, target_weights, weights, tau):
#         for (target_w, w) in zip(target_weights, weights):
#             target_w.assign(tau * w + (1 - tau) * target_w)
#
#     def train(self, initial_episode, episodes, metrics=(), batch_size=None):
#         for metric in metrics: metric.on_task_begin()
#
#         for i in tqdm(range(initial_episode, initial_episode+episodes), desc="Episode"):
#             episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
#             for metric in metrics: metric.on_episode_begin(episode_data)
#
#             self.env.reset()
#             current_observations, _, _, _ = self.env.observe()
#             while not self.env.is_episode_finished():
#                 actions, action_values, explored = self.get_action(current_observations, self.exploration)
#                 self.env.take_action(actions)
#                 next_observations, rewards, terminated, frame = self.env.observe()
#
#                 self.replay_buffer.insert_transition([current_observations, actions, rewards, next_observations, terminated])
#
#                 current_observations = next_observations
#
#                 step_data = {
#                     "current_obs": current_observations,
#                     "action_value": action_values,
#                     "action": actions,
#                     "reward": rewards,
#                     "next_obs": next_observations,
#                     "explored": explored,
#                     "frame": frame
#                 }
#                 for metric in metrics: metric.on_episode_step(step_data)
#
#                 if self.step_counter % self.learn_after_steps == 0:
#                     current_observations_batch, actions_batch, rewards_batch, next_observations_batch, _ = self.replay_buffer.sample_batch_transitions(batch_size=batch_size)
#
#                     all_agent_current_observations = tf.stack([current_observations_batch[agent] for agent in self.agents], axis=1)
#                     all_agent_actions = tf.stack([actions_batch[agent] for agent in self.agents], axis=1)
#                     all_agent_next_observations = tf.stack([next_observations_batch[agent] for agent in self.agents], axis=1)
#                     all_agent_next_actions = tf.stack([self.actor_networks[agent](current_observations_batch[agent]) for agent in self.agents], axis=1)
#                     all_adversary_current_observations = tf.stack([current_observations_batch[agent] for agent in self.adversaries],
#                                                               axis=1)
#                     all_adversary_actions = tf.stack([actions_batch[agent] for agent in self.adversaries], axis=1)
#                     all_adversary_next_observations = tf.stack([next_observations_batch[agent] for agent in self.adversaries], axis=1)
#                     all_adversary_next_actions = tf.stack(
#                         [self.actor_networks[agent](current_observations_batch[agent]) for agent in self.adversaries], axis=1)
#
#                     for agent_num, agent in enumerate(self.agents):
#                         # Reusing a single graph for each agent
#                         set_weights(self.current_actor_network, get_weights(self.actor_networks[agent]))
#                         set_weights(self.current_critic_network, get_weights(self.critic_networks[agent]))
#                         set_weights(self.current_target_actor_network, get_weights(self.target_actor_networks[agent]))
#                         set_weights(self.current_target_critic_network, get_weights(self.target_critic_networks[agent]))
#                         self.current_actor_network.optimizer = self.actor_networks[agent].optimizer
#                         self.current_critic_network.optimizer = self.critic_networks[agent].optimizer
#
#                         if all_agent_current_observations.shape[0] >= batch_size:
#                             self._critic_agent_train_step(all_agent_current_observations, all_agent_actions, rewards_batch[agent], all_agent_next_observations, all_agent_next_actions)
#                             self._actor_agent_train_step(all_agent_current_observations, all_agent_actions, current_observations_batch[agent], tf.convert_to_tensor(agent_num))
#                             set_weights(self.actor_networks[agent], get_weights(self.current_actor_network))
#                             set_weights(self.critic_networks[agent], get_weights(self.current_critic_network))
#                             set_weights(self.target_actor_networks[agent], get_weights(self.current_target_actor_network))
#                             set_weights(self.target_critic_networks[agent], get_weights(self.current_target_critic_network))
#                             self.update_targets(self.target_actor_networks[agent].trainable_variables, self.actor_networks[agent].trainable_variables, self.tau)
#                             self.update_targets(self.target_critic_networks[agent].trainable_variables, self.critic_networks[agent].trainable_variables, self.tau)
#                     for agent_num, agent in enumerate(self.adversaries):
#                         # Reusing a single graph for each agent
#                         set_weights(self.current_actor_adversary_network, get_weights(self.actor_networks[agent]))
#                         set_weights(self.current_critic_adversary_network, get_weights(self.critic_networks[agent]))
#                         set_weights(self.current_target_actor_adversary_network, get_weights(self.target_actor_networks[agent]))
#                         set_weights(self.current_target_critic_adversary_network, get_weights(self.target_critic_networks[agent]))
#                         self.current_actor_adversary_network.optimizer = self.actor_networks[agent].optimizer
#                         self.current_critic_adversary_network.optimizer = self.critic_networks[agent].optimizer
#
#                         if all_adversary_current_observations.shape[0] >= batch_size:
#                             self._critic_adversary_train_step(all_adversary_current_observations, all_adversary_actions, rewards_batch[agent],
#                                                     all_adversary_next_observations, all_adversary_next_actions)
#                             self._actor_adversary_train_step(all_adversary_current_observations, all_adversary_actions, current_observations_batch[agent], tf.convert_to_tensor(agent_num))
#                             set_weights(self.actor_networks[agent], get_weights(self.current_actor_adversary_network))
#                             set_weights(self.critic_networks[agent], get_weights(self.current_critic_adversary_network))
#                             set_weights(self.target_actor_networks[agent],
#                                         get_weights(self.current_target_actor_adversary_network))
#                             set_weights(self.target_critic_networks[agent],
#                                         get_weights(self.current_target_critic_adversary_network))
#                             self.update_targets(self.target_actor_networks[agent].trainable_variables, self.actor_networks[agent].trainable_variables, self.tau)
#                             self.update_targets(self.target_critic_networks[agent].trainable_variables, self.critic_networks[agent].trainable_variables,
#                                                 self.tau)
#
#                 self.step_counter += 1
#             episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
#             for metric in metrics: metric.on_episode_end(episode_data)
#
#             if (i + 1) % self.exploration_decay_after == 0:
#                 self.exploration = min(self.min_exploration, self.exploration/self.exploration_decay)
#         for metric in metrics: metric.on_task_end()
#
#     def get_action(self, observations, explore=0.0):
#         actions = {}
#         action_values = {}
#         exploreds = {}
#         agent_observations = tf.convert_to_tensor([[observations[agent] for agent in self.agents]])
#         adversary_observations = tf.convert_to_tensor([[observations[agent] for agent in self.adversaries]])
#         for agent, observation in observations.items():
#             action = self.actor_networks[agent](tf.convert_to_tensor([observation], dtype=tf.float32))
#             explored = tf.convert_to_tensor(False)
#             if tf.random.uniform(shape=(), maxval=1) < explore:
#                 action = action + tf.convert_to_tensor([self.env.get_random_action(agent)], tf.float32)
#                 explored = tf.convert_to_tensor(True)
#             actions[agent] = action[0]
#             exploreds[agent] = explored
#         agent_actions = tf.convert_to_tensor([[actions[agent] for agent in self.agents]])
#         adversary_actions = tf.convert_to_tensor([[actions[agent] for agent in self.adversaries]])
#         for agent in observations.keys():
#             if agent in self.agents:
#                 value = self.critic_networks[agent]([agent_observations, agent_actions])
#             else:
#                 value = self.critic_networks[agent]([adversary_observations, adversary_actions])
#             action_values[agent] = value[0]
#         return actions, action_values, exploreds
#
#     def infer(self, initial_episode, episodes, metrics: list[Metric] = (), exploration=0.0):
#         for metric in metrics: metric.on_task_begin()
#
#         for ep in range(initial_episode, initial_episode+episodes):
#             episode_data = {"episode": ep, "exploration": exploration, "step": self.step_counter}
#             for metric in metrics: metric.on_episode_begin(episode_data)
#
#             self.env.reset()
#             current_observations, rewards, terminated, frame = self.env.observe()
#             while not self.env.is_episode_finished():
#                 actions, action_values, explored = self.get_action(current_observations, exploration)
#                 self.env.take_action(actions)
#                 next_observations, rewards, terminated, frame = self.env.observe()
#
#                 step_data = {
#                     "current_obs":current_observations,
#                     "action_value": action_values,
#                     "action": actions,
#                     "reward": rewards,
#                     "next_obs": next_observations,
#                     "explored": explored,
#                     "frame": frame
#                 }
#                 for metric in metrics: metric.on_episode_step(step_data)
#                 current_observations = next_observations
#             episode_data = {"episode_data": ep, "exploration": exploration, "step": self.step_counter}
#             for metric in metrics: metric.on_episode_end(episode_data)
#         for metric in metrics: metric.on_task_end()
#
#     def save(self, path=""):
#         super().save(path)
#
#     def load(self, path=""):
#         super().load(path)

class MADDPG(DriverAlgorithm):

    def __init__(self, env: MAParallelEnvironment, agents, adversaries, actor_network=None, actor_adversary_network=None, critic_network=None, learn_after_steps=1,
                 replay_size=1000, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
                 exploration_decay_after=100, discount_factor=0.9, tau=0.001):
        super().__init__()
        self.env = env
        self.agents = agents
        self.adversaries = adversaries

        if actor_network and critic_network:
            self.actor_networks = {agent: clone_model(actor_network) for agent in self.agents}
            self.actor_networks.update({agent: clone_model(actor_adversary_network) for agent in self.adversaries})
            for an in self.actor_networks.values():
                an.compile(optimizer=Adam.from_config(actor_network.optimizer.get_config()))
            self.target_actor_networks = {agent: clone_model(actor_network) for agent in self.agents}
            self.target_actor_networks.update({agent: clone_model(actor_adversary_network) for agent in self.adversaries})
            self.critic_networks = {agent: clone_model(critic_network) for agent in self.agents+self.adversaries}
            for cn in self.critic_networks.values():
                cn.compile(optimizer=Adam.from_config(critic_network.optimizer.get_config()))
            self.target_critic_networks = {agent: clone_model(critic_network) for agent in self.agents+self.adversaries}
            self.current_actor_network, self.current_critic_network = clone_model(actor_network), clone_model(critic_network)
            self.current_target_actor_network, self.current_target_critic_network = clone_model(actor_network), clone_model(critic_network)
            self.current_actor_adversary_network = clone_model(actor_adversary_network)
            self.current_target_actor_adversary_network = clone_model(actor_adversary_network)
        else:
            self.actor_networks = None
            self.target_actor_networks = None
            self.critic_networks = None
            self.target_critic_networks = None
            self.current_actor_network, self.current_critic_network = None, None
            self.current_target_actor_network, self.current_target_critic_network = None, None
            self.current_actor_adversary_network = None
            self.current_target_actor_adversary_network = None

        self.learn_after_steps = learn_after_steps
        self.replay_buffer = MAExperienceReplay(agents=self.agents+self.adversaries, max_transitions=replay_size, continuous=True)
        self.discount_factor = tf.convert_to_tensor(discount_factor)
        self.exploration = exploration
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.tau = tf.convert_to_tensor(tau)
        self.step_counter = 1
        self.critic_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def _critic_agent_train_step(self, current_agent_observations, actions_agent, current_adversary_observations, actions_adversary, rewards, next_agent_observations, next_actions_agent, next_adversary_observations, next_actions_adversary):
        targets = tf.expand_dims(rewards, axis=1) + self.discount_factor * self.current_target_critic_network([next_agent_observations, next_actions_agent, next_adversary_observations, next_actions_adversary])

        with tf.GradientTape() as tape:
            critic_value = self.current_critic_network([current_agent_observations, actions_agent, current_adversary_observations, actions_adversary])
            critic_loss = self.critic_loss(targets, critic_value)

        critic_grads = tape.gradient(critic_loss, self.current_critic_network.trainable_variables)
        self.current_critic_network.optimizer.apply_gradients(zip(critic_grads, self.current_critic_network.trainable_variables))

    @tf.function
    def _actor_agent_train_step(self, current_agent_observations, actions_agent, current_adversary_observations, actions_adversary, current_observation, agent):
        indices = tf.expand_dims(tf.range(current_agent_observations.shape[0]), axis=1)
        filler = tf.fill(dims=(current_agent_observations.shape[0], 1), value=agent)
        indices = tf.concat([indices, filler], axis=1)
        with tf.GradientTape() as tape:
            new_actions = tf.tensor_scatter_nd_update(actions_agent, indices, self.current_actor_network(current_observation))
            actor_loss = -tf.reduce_mean(self.current_critic_network([current_agent_observations, new_actions, current_adversary_observations, actions_adversary]))
        actor_grads = tape.gradient(actor_loss, self.current_actor_network.trainable_variables)
        self.current_actor_network.optimizer.apply_gradients(zip(actor_grads, self.current_actor_network.trainable_variables))

    @tf.function
    def _actor_adversary_train_step(self, current_agent_observations, actions_agent, current_adversary_observations, actions_adversary, current_observation, agent):
        indices = tf.expand_dims(tf.range(current_agent_observations.shape[0]), axis=1)
        filler = tf.fill(dims=(current_agent_observations.shape[0], 1), value=agent)
        indices = tf.concat([indices, filler], axis=1)
        with tf.GradientTape() as tape:
            new_actions = tf.tensor_scatter_nd_update(actions_adversary, indices,
                                                      self.current_actor_adversary_network(current_observation))
            actor_loss = -tf.reduce_mean(self.current_critic_network([current_agent_observations, actions_agent, current_adversary_observations, new_actions]))
        actor_grads = tape.gradient(actor_loss, self.current_actor_adversary_network.trainable_variables)
        self.current_actor_adversary_network.optimizer.apply_gradients(
            zip(actor_grads, self.current_actor_adversary_network.trainable_variables))

    @tf.function
    def update_targets(self, target_weights, weights, tau):
        for (target_w, w) in zip(target_weights, weights):
            target_w.assign(tau * w + (1 - tau) * target_w)

    def train(self, initial_episode, episodes, metrics=(), batch_size=None):
        for metric in metrics: metric.on_task_begin()

        for i in tqdm(range(initial_episode, initial_episode+episodes), desc="Episode"):
            episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_begin(episode_data)

            self.env.reset()
            current_observations, _, _, _ = self.env.observe()
            while not self.env.is_episode_finished():
                actions, action_values, explored = self.get_action(current_observations, self.exploration)
                self.env.take_action(actions)
                next_observations, rewards, terminated, frame = self.env.observe()

                self.replay_buffer.insert_transition([current_observations, actions, rewards, next_observations, terminated])

                current_observations = next_observations

                step_data = {
                    "current_obs": current_observations,
                    "action_value": action_values,
                    "action": actions,
                    "reward": rewards,
                    "next_obs": next_observations,
                    "explored": explored,
                    "frame": frame
                }
                for metric in metrics: metric.on_episode_step(step_data)

                if self.step_counter % self.learn_after_steps == 0:
                    current_observations_batch, actions_batch, rewards_batch, next_observations_batch, _ = self.replay_buffer.sample_batch_transitions(batch_size=batch_size)

                    all_agent_current_observations = tf.stack([current_observations_batch[agent] for agent in self.agents], axis=1)
                    all_agent_actions = tf.stack([actions_batch[agent] for agent in self.agents], axis=1)
                    all_agent_next_observations = tf.stack([next_observations_batch[agent] for agent in self.agents], axis=1)
                    all_agent_next_actions = tf.stack([self.actor_networks[agent](current_observations_batch[agent]) for agent in self.agents], axis=1)
                    all_adversary_current_observations = tf.stack([current_observations_batch[agent] for agent in self.adversaries],
                                                              axis=1)
                    all_adversary_actions = tf.stack([actions_batch[agent] for agent in self.adversaries], axis=1)
                    all_adversary_next_observations = tf.stack([next_observations_batch[agent] for agent in self.adversaries], axis=1)
                    all_adversary_next_actions = tf.stack(
                        [self.actor_networks[agent](current_observations_batch[agent]) for agent in self.adversaries], axis=1)

                    for agent_num, agent in enumerate(self.agents):
                        # Reusing a single graph for each agent
                        set_weights(self.current_actor_network, get_weights(self.actor_networks[agent]))
                        set_weights(self.current_critic_network, get_weights(self.critic_networks[agent]))
                        set_weights(self.current_target_actor_network, get_weights(self.target_actor_networks[agent]))
                        set_weights(self.current_target_critic_network, get_weights(self.target_critic_networks[agent]))
                        self.current_actor_network.optimizer = self.actor_networks[agent].optimizer
                        self.current_critic_network.optimizer = self.critic_networks[agent].optimizer

                        if all_agent_current_observations.shape[0] >= batch_size:
                            self._critic_agent_train_step(all_agent_current_observations, all_agent_actions, all_adversary_current_observations, all_adversary_actions, rewards_batch[agent], all_agent_next_observations, all_agent_next_actions, all_adversary_next_observations, all_adversary_next_actions)
                            self._actor_agent_train_step(all_agent_current_observations, all_agent_actions, all_adversary_current_observations, all_adversary_actions, current_observations_batch[agent], tf.convert_to_tensor(agent_num))
                            set_weights(self.actor_networks[agent], get_weights(self.current_actor_network))
                            set_weights(self.critic_networks[agent], get_weights(self.current_critic_network))
                            set_weights(self.target_actor_networks[agent], get_weights(self.current_target_actor_network))
                            set_weights(self.target_critic_networks[agent], get_weights(self.current_target_critic_network))
                            self.update_targets(self.target_actor_networks[agent].trainable_variables, self.actor_networks[agent].trainable_variables, self.tau)
                            self.update_targets(self.target_critic_networks[agent].trainable_variables, self.critic_networks[agent].trainable_variables, self.tau)
                    for agent_num, agent in enumerate(self.adversaries):
                        # Reusing a single graph for each agent
                        set_weights(self.current_actor_adversary_network, get_weights(self.actor_networks[agent]))
                        set_weights(self.current_critic_network, get_weights(self.critic_networks[agent]))
                        set_weights(self.current_target_actor_adversary_network, get_weights(self.target_actor_networks[agent]))
                        set_weights(self.current_target_critic_network, get_weights(self.target_critic_networks[agent]))
                        self.current_actor_adversary_network.optimizer = self.actor_networks[agent].optimizer
                        self.current_critic_network.optimizer = self.critic_networks[agent].optimizer

                        if all_adversary_current_observations.shape[0] >= batch_size:
                            self._critic_agent_train_step(all_agent_current_observations, all_agent_actions, all_adversary_current_observations, all_adversary_actions, rewards_batch[agent],
                                                    all_agent_next_observations, all_agent_actions, all_adversary_next_observations, all_adversary_next_actions)
                            self._actor_adversary_train_step(all_agent_current_observations, all_agent_actions, all_adversary_current_observations, all_adversary_actions, current_observations_batch[agent], tf.convert_to_tensor(agent_num))
                            set_weights(self.actor_networks[agent], get_weights(self.current_actor_adversary_network))
                            set_weights(self.critic_networks[agent], get_weights(self.current_critic_network))
                            set_weights(self.target_actor_networks[agent],
                                        get_weights(self.current_target_actor_adversary_network))
                            set_weights(self.target_critic_networks[agent],
                                        get_weights(self.current_target_critic_network))
                            self.update_targets(self.target_actor_networks[agent].trainable_variables, self.actor_networks[agent].trainable_variables, self.tau)
                            self.update_targets(self.target_critic_networks[agent].trainable_variables, self.critic_networks[agent].trainable_variables,
                                                self.tau)

                self.step_counter += 1
            episode_data = {"episode": i, "exploration": self.exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_end(episode_data)

            if (i + 1) % self.exploration_decay_after == 0:
                self.exploration = min(self.min_exploration, self.exploration/self.exploration_decay)
        for metric in metrics: metric.on_task_end()

    def get_action(self, observations, explore=0.0):
        actions = {}
        action_values = {}
        exploreds = {}
        agent_observations = tf.convert_to_tensor([[observations[agent] for agent in self.agents]])
        adversary_observations = tf.convert_to_tensor([[observations[agent] for agent in self.adversaries]])
        for agent, observation in observations.items():
            action = self.actor_networks[agent](tf.convert_to_tensor([observation], dtype=tf.float32))
            explored = tf.convert_to_tensor(False)
            if tf.random.uniform(shape=(), maxval=1) < explore:
                action = action + tf.convert_to_tensor([self.env.get_random_action(agent)], tf.float32)
                explored = tf.convert_to_tensor(True)
            actions[agent] = action[0]
            exploreds[agent] = explored
        agent_actions = tf.convert_to_tensor([[actions[agent] for agent in self.agents]])
        adversary_actions = tf.convert_to_tensor([[actions[agent] for agent in self.adversaries]])
        for agent in observations.keys():
            value = self.critic_networks[agent]([agent_observations, agent_actions, adversary_observations, adversary_actions])
            action_values[agent] = value[0]
        return actions, action_values, exploreds

    def infer(self, initial_episode, episodes, metrics: list[Metric] = (), exploration=0.0):
        for metric in metrics: metric.on_task_begin()

        for ep in range(initial_episode, initial_episode+episodes):
            episode_data = {"episode": ep, "exploration": exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_begin(episode_data)

            self.env.reset()
            current_observations, rewards, terminated, frame = self.env.observe()
            while not self.env.is_episode_finished():
                actions, action_values, explored = self.get_action(current_observations, exploration)
                self.env.take_action(actions)
                next_observations, rewards, terminated, frame = self.env.observe()

                step_data = {
                    "current_obs":current_observations,
                    "action_value": action_values,
                    "action": actions,
                    "reward": rewards,
                    "next_obs": next_observations,
                    "explored": explored,
                    "frame": frame
                }
                for metric in metrics: metric.on_episode_step(step_data)
                current_observations = next_observations
            episode_data = {"episode_data": ep, "exploration": exploration, "step": self.step_counter}
            for metric in metrics: metric.on_episode_end(episode_data)
        for metric in metrics: metric.on_task_end()

    def save(self, path=""):
        for agent, actor_network in self.actor_networks.items():
            actor_network.save(os.path.join(path, agent+"_actor_network"))
        for agent, actor_target_network in self.target_actor_networks.items():
            actor_target_network.save(os.path.join(path, agent+"_actor_target_network"))
        for agent, critic_network in self.critic_networks.items():
            critic_network.save(os.path.join(path, agent+"_critic_network"))
        for agent, critic_target_network in self.target_critic_networks.items():
            critic_target_network.save(os.path.join(path, agent+"_critic_target_network"))
        self.replay_buffer.save(os.path.join(path, "replay"))

    def load(self, path=""):
        self.actor_networks={}
        self.critic_networks = {}
        self.target_actor_networks={}
        self.target_critic_networks={}
        for agent in self.agents+self.adversaries:
            self.actor_networks[agent] = load_model(os.path.join(path, agent+"_actor_network"))
            self.critic_networks[agent] = load_model(os.path.join(path, agent+"_critic_network"))
            try:
                self.target_actor_networks[agent] = load_model(os.path.join(path, agent+"_actor_target_network"))
            except:
                if self.actor_networks[agent] is not None:
                    self.target_actor_networks[agent] = clone_model(self.actor_networks[agent])
            try:
                self.target_critic_networks[agent] = load_model(os.path.join(path, agent+"_critic_target_network"))
            except:
                if self.critic_networks[agent] is not None:
                    self.target_critic_networks[agent] = clone_model(self.critic_networks[agent])
        self.replay_buffer.load(os.path.join(path, "replay"))


class MAExperienceReplay(ReplayBuffer):

    def __init__(self, agents, max_transitions=1000, continuous=False):
        self.agents = agents
        self.max_transitions = max_transitions
        self.continuous = continuous
        self.current_observations = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for agent in self.agents}
        if self.continuous:
            self.actions = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for agent in self.agents}
        else:
            self.actions = {agent: tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False) for agent in self.agents}
        self.rewards = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for agent in self.agents}
        self.next_observations = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for agent in self.agents}
        self.terminals = {agent: tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False) for agent in self.agents}
        self.current_index = 0

    def _insert_transition_at(self, transition, index, agent):
        self.current_observations[agent] = self.current_observations[agent].write(index, transition[0][agent])
        self.actions[agent] = self.actions[agent].write(index, transition[1][agent])
        self.rewards[agent] = self.rewards[agent].write(index, transition[2][agent])
        self.next_observations[agent] = self.next_observations[agent].write(index, transition[3][agent])
        self.terminals[agent] = self.terminals[agent].write(index, transition[4][agent])

    def insert_transition(self, transition):
        for agent in self.agents:
            self._insert_transition_at(transition, self.current_index % self.max_transitions, agent)
        self.current_index += 1

    def sample_batch_transitions(self, batch_size=16):
        buf_len = self.current_observations[self.agents[0]].size()
        if buf_len <= batch_size:
            sampled_indices = tf.random.uniform(shape=(buf_len,), maxval=buf_len, dtype=tf.int32)
        else:
            sampled_indices = tf.random.uniform(shape=(batch_size,), maxval=buf_len, dtype=tf.int32)

        return {agent: self.current_observations[agent].gather(sampled_indices) for agent in self.agents}, \
            {agent: self.actions[agent].gather(sampled_indices) for agent in self.agents}, \
            {agent: self.rewards[agent].gather(sampled_indices) for agent in self.agents}, \
            {agent: self.next_observations[agent].gather(sampled_indices) for agent in self.agents}, \
            {agent: self.terminals[agent].gather(sampled_indices) for agent in self.agents}

    def save(self, path=""):
        for agent, current_obs in self.current_observations.items():
            tf.io.write_file(os.path.join(path, agent+"_current_obs.tfw"), tf.io.serialize_tensor(current_obs.stack()))
        for agent, actions in self.actions.items():
            tf.io.write_file(os.path.join(path, agent+"_actions.tfw"), tf.io.serialize_tensor(actions.stack()))
        for agent, rewards in self.rewards.items():
            tf.io.write_file(os.path.join(path, agent+"_rewards.tfw"), tf.io.serialize_tensor(rewards.stack()))
        for agent, next_obs in self.next_observations.items():
            tf.io.write_file(os.path.join(path, agent+"_next_states.tfw"), tf.io.serialize_tensor(next_obs.stack()))
        for agent, terminals in self.terminals.items():
            tf.io.write_file(os.path.join(path, agent+"_terminals.tfw"), tf.io.serialize_tensor(terminals.stack()))

    def load(self, path=""):
        self.current_observations = {
            agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for agent in
            self.agents}
        if self.continuous:
            self.actions = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for
                            agent in self.agents}
        else:
            self.actions = {agent: tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False) for agent
                            in self.agents}
        self.rewards = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False) for agent
                        in self.agents}
        self.next_observations = {agent: tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                                  for agent in self.agents}
        self.terminals = {agent: tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False) for agent in
                          self.agents}
        self.current_index = 0

        for agent in self.agents:
            try:
                self.current_observations[agent] = self.current_observations[agent].unstack(
                    tf.io.parse_tensor(tf.io.read_file(os.path.join(path, agent+"_current_obs.tfw")), tf.float32))

                if self.continuous:
                    self.actions[agent] = self.actions[agent].unstack(
                        tf.io.parse_tensor(tf.io.read_file(os.path.join(path, agent+"_actions.tfw")), tf.float32))
                else:
                    self.actions[agent] = self.actions[agent].unstack(
                        tf.io.parse_tensor(tf.io.read_file(os.path.join(path, agent+"_actions.tfw")), tf.int32))
                self.rewards[agent] = self.rewards[agent].unstack(
                    tf.io.parse_tensor(tf.io.read_file(os.path.join(path, agent+"_rewards.tfw")), tf.float32))
                self.next_observations[agent] = self.next_observations[agent].unstack(
                    tf.io.parse_tensor(tf.io.read_file(os.path.join(path, agent+"_next_obs.tfw")), tf.float32))
                self.terminals[agent] = self.terminals[agent].unstack(
                    tf.io.parse_tensor(tf.io.read_file(os.path.join(path, agent+"_terminals.tfw")), tf.bool))
                print("Found", self.current_observations[self.agents[0]].size().numpy(), "transitions")
            except Exception as e:
                print(e)
                print("No Experience Replay found")

class MATotalRewardMetric(Metric):
    # Tracks the total reward in an episode

    def __init__(self, path=""):
        super(MATotalRewardMetric, self).__init__(path)
        self.name = "Total Reward"
        self.episodic_data = {"episode": [], "total_reward": [], "step": []}
        self.total_reward = 0

    def on_task_begin(self, data=None):
        self.load()

    def on_episode_begin(self, data=None):
        self.total_reward = 0

    def on_episode_step(self, data=None):
        self.total_reward += np.mean([data["reward"][agent] for agent in data["reward"].keys()])

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