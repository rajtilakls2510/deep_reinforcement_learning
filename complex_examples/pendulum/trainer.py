import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import AvgTotalReward
from pendulum_env_wrapper import PendulumEnvironment

env = PendulumEnvironment(gym.make("Pendulum-v1", render_mode = "rgb_array"))
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

metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"), continuous=True)

driver_algorithm = DeepDPG(
    actor_network,
    critic_network,
    learn_after_steps=1,
    replay_size=1_00_000,
    discount_factor=0.99,
    tau=0.005
)
agent = Agent(env, driver_algorithm)
# 1_000 episodes
for i in range(2):
    print("Training Iteration:", i)
    agent.train(initial_episode=100 * i, episodes=100, metric=metric, batch_size=64)
    agent.save(AGENT_PATH)
env.close()

# Load agent and train
# driver_algorithm = DeepDPG(
#     learn_after_steps=4,
#     replay_size=1_00_000,
#     discount_factor=0.99,
#     tau=0.001
# )
#
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(4, 1_00):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
#     agent.save(AGENT_PATH)
# env.close()
