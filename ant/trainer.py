import gymnasium as gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG, TD3
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, AbsoluteValueErrorMetric
from ant_env_wrappers import AntEnvironment
import tensorflow as tf
import numpy as np

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

tf.random.set_seed(tf.random.uniform(shape=(1,), minval=0, maxval=1000, dtype=tf.int32))
np.random.seed(np.random.randint(0, 1000))

# Wrapping the gym environment to interface with our library
env = AntEnvironment(gym.make("Hopper-v4", render_mode="rgb_array", terminate_when_unhealthy=False))
AGENT_PATH = "hopper_agent"

# Actor Network
state_input = Input(shape=(11,))
x = Dense(400, activation="relu")(state_input)
x = BatchNormalization()(x)
x = Dense(300, activation="relu")(x)
x = BatchNormalization()(x)
# x = Dense(256)(x)
# x = BatchNormalization()(x)
# x = LeakyReLU(alpha=0.0)(x)
output = Dense(3, activation="tanh", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
actor_network = Model(inputs=state_input, outputs=output)
actor_network.compile(optimizer=Adam(learning_rate=0.001))

# Critic Network
state_input = Input(shape=(11,))
x1 = Dense(400, activation="relu")(state_input)
x1 = BatchNormalization()(x1)

action_input = Input(shape=(3,))
x2 = Dense(400, activation="relu")(action_input)

x = Add()([x1, x2])
x = Dense(300, activation="relu")(x)
output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
critic_network1 = Model(inputs=[state_input, action_input], outputs=output)
critic_network1.compile(optimizer=Adam(learning_rate=0.001))

critic_network2 = Model.from_config(critic_network1.get_config())
critic_network2.compile(optimizer=Adam(learning_rate=0.001))

# Setting up metrics for training
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
# avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
# value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))

total_reward_eval = TotalRewardMetric(os.path.join(AGENT_PATH, "eval_metric"))
ep_length_eval = EpisodeLengthMetric(os.path.join(AGENT_PATH, "eval_metric"))

# =============== Starting training from scratch ======================

# Creating the algorithm that will be used to train the agent
driver_algorithm = TD3(
    actor_network,
    critic_network1,
    critic_network2,
    learn_after_steps=2,
    replay_size=500_000,
    min_replay_size=1_000,
    discount_factor=0.99,
    exploration=1,
    exploration_decay=1,
    tau=0.005,
    target_noise_std=0.2,
    target_noise_clipvalue=0.5
)
# Creating the Agent class
agent = Agent(env, driver_algorithm)

# Training for 2000 episodes. Saving the agent every 1 episode
for i in range(2000):
    print("Training Iteration:", i)
    agent.train(
        initial_episode=5 * i,
        episodes=5,
        metrics=[ep_length, total_reward],
        batch_size=100)
    if i % 10 == 0:
        agent.save(AGENT_PATH)
    agent.evaluate(initial_episode=5 * i, episodes=1, metrics=[total_reward_eval, ep_length_eval], exploration=0.0)
env.close()

# ============================== Loading the agent and resuming training ================
# driver_algorithm = DeepDPG(
#     learn_after_steps=3,
#     replay_size=1_00_000,
#     exploration = 1,
#     exploration_decay = 1,
#     discount_factor=0.99,
#     tau=0.001,
#     step=268695
# )
#
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# # Resuming training from 258th episode and training till 1_000th episode
# for i in range(275, 500):
#     print("Training Iteration: ", i)
#     agent.train(
#     initial_episode=1 * i,
#     episodes=1,
#     metrics=[ep_length, total_reward, avg_q, value_error],
#     batch_size=64)
#     agent.save(AGENT_PATH)
#     agent.evaluate(initial_episode=1 * i, episodes=1, metrics=[total_reward_eval, ep_length_eval], exploration=0.0)
# env.close()
