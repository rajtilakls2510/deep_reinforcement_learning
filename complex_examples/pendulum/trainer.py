import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, AbsoluteValueErrorMetric
from pendulum_env_wrapper import PendulumEnvironment

# Wrapping the gym environment to interface with our library
env = PendulumEnvironment(gym.make("Pendulum-v1", render_mode="rgb_array"))
AGENT_PATH = "pendulum_agent"

# Actor Network
state_input = Input(shape=(3,))
x = Dense(256, activation="relu")(state_input)
x = Dense(256, activation="relu")(x)
output = Dense(1, activation="tanh", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
actor_network = Model(inputs=state_input, outputs=output)
actor_network.compile(optimizer=Adam(learning_rate=0.001))

# Critic Network
state_input = Input(shape=(3,))
x1 = Dense(16, activation='relu')(state_input)
x1 = Dense(32, activation='relu')(x1)

action_input = Input(shape=(1,))
x2 = Dense(32, activation='relu')(action_input)

x = Concatenate()([x1, x2])
x = Dense(256, activation="relu")(x)
x = Dense(256, activation="relu")(x)
output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
critic_network = Model(inputs=[state_input, action_input], outputs=output)
critic_network.compile(optimizer=Adam(learning_rate=0.002))

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
    learn_after_steps=1,
    replay_size=1_00_000,
    discount_factor=0.99,
    tau=0.005
)
# Creating the Agent class
agent = Agent(env, driver_algorithm)

# Training for 120 episodes. Saving the agent every 10 episodes
for i in range(12):
    print("Training Iteration:", i)
    agent.train(
        initial_episode=10 * i,
        episodes=10,
        metrics=[ep_length, total_reward, avg_q, value_error],
        batch_size=64
    )
    agent.save(AGENT_PATH)
env.close()

# ============================== Loading the agent and resuming training ================
# driver_algorithm = DeepDPG(
#     learn_after_steps=4,
#     replay_size=1_00_000,
#     discount_factor=0.99,
#     tau=0.001
# )
#
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# # Resuming training from 40th episode and training till 1_000th episode
# for i in range(10, 12):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metrics=[ep_length, total_reward, avg_q, value_error], batch_size=64)
#     agent.save(AGENT_PATH)
# env.close()
