import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import AvgTotalReward
from interface import LunarLanderContinuousTerminal, LunarLanderContinuousInterpreter

interpreter = LunarLanderContinuousInterpreter(
    LunarLanderContinuousTerminal(gym.make("LunarLander-v2", continuous=True)))
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

metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"))

# Start training from scratch
# driver_algorithm = DeepDPG(
#     actor_network,
#     critic_network,
#     learn_after_steps=4,
#     discount_factor=0.99,
#     exploration=1.0,
#     min_exploration=0.01,
#     exploration_decay=1,
#     exploration_decay_after=1,
#     replay_size=1_00_000,
#     tau=0.001
# )
#
# agent = Agent(interpreter, driver_algorithm)
# for i in range(2_00):
#     print("Training Iteration:", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
#     agent.save(AGENT_PATH)
# interpreter.close()

# Load agent and train (change exploration param)
driver_algorithm = DeepDPG(
    learn_after_steps=4,
    discount_factor=0.99,
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1,
    exploration_decay_after=1,
    replay_size=1_00_000,
    tau=0.001
)

agent = Agent(interpreter, driver_algorithm)
agent.load(AGENT_PATH)
for i in range(1_82, 2_00):
    print("Training Iteration: ", i)
    agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
    agent.save(AGENT_PATH)
interpreter.close()
