import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, AbsoluteValueErrorMetric
from lunarlandercont_env_wrapper import LunarLanderContinuousEnvironment

# Wrapping the gym environment to interface with our library
env = LunarLanderContinuousEnvironment(gym.make("LunarLander-v2", continuous=True, render_mode = "rgb_array"))
AGENT_PATH = "lunar_lander_cont_agent"

# Actor Network
state_input = Input(shape=(8,))
x = Dense(256, activation="relu")(state_input)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
output = Dense(2, activation="tanh", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
actor_network = Model(inputs=state_input, outputs=output)
actor_network.compile(optimizer=Adam(learning_rate=0.0005))

# Critic Network
state_input = Input(shape=(8,))
x1 = Dense(192, activation="relu")(state_input)
x1 = BatchNormalization()(x1)

action_input = Input(shape=(2,))
x2 = Dense(64, activation="relu")(action_input)

x = Concatenate()([x1, x2])
x = Dense(256, activation="relu")(x)
x = Dense(256, activation="relu")(x)
output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
critic_network = Model(inputs=[state_input, action_input], outputs=output)
critic_network.compile(optimizer=Adam(learning_rate=0.001))

# Setting up metrics for training
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))

# =============== Starting training from scratch ======================

# Creating the algorithm that will be used to train the agent
driver_algorithm = DeepDPG(
    actor_network,
    critic_network,
    learn_after_steps=4,
    discount_factor=0.99,
    exploration=1.0,
    min_exploration=0.01,
    exploration_decay=1,
    exploration_decay_after=1,
    replay_size=1_00_000,
    tau=0.001
)
# Creating the Agent class
agent = Agent(env, driver_algorithm)

# Training for 2_000 episodes. Saving the agent every 100 episodes
for i in range(2_0):
    print("Training Iteration: ", i)
    agent.train(
        initial_episode=100 * i,
        episodes=100,
        batch_size=64,
        metrics=[ep_length, total_reward, avg_q, value_error]
    )
    agent.save(AGENT_PATH)
env.close()

# ============================== Loading the agent and resuming training ================
# driver_algorithm = DeepDPG(
#     learn_after_steps=4,
#     discount_factor=0.99,
#     exploration=1,
#     min_exploration=0.01,
#     exploration_decay=1,
#     exploration_decay_after=1,
#     replay_size=1_00_000,
#     tau=0.001
# )
#
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# # Resuming training from 200th episode and training till 1_000th episode
# for i in range(2, 1_0):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, batch_size=64, metrics=[ep_length, total_reward, avg_q, value_error])
#     agent.save(AGENT_PATH)
# env.close()
