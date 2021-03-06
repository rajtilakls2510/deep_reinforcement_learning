import os
import gym
from interface import CartpoleTerminal, CartpoleInterpreter
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

env = gym.make("CartPole-v1")
interpreter = CartpoleInterpreter(CartpoleTerminal(env))

AGENT_PATH = "cart_pole_agent3"

# Q Network

net_input = Input(shape=(4,))
x = Dense(32, activation="relu")(net_input)
x = Dense(16, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

optimizer = Adam()
q_net.compile(optimizer=optimizer)
metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"))

# # Start training from scratch
driver_algorithm = DeepQLearning(
    q_net,
    learn_after_steps=3,
    replay_size= 1_00_000,
    discount_factor= 0.99,
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1.005,
    exploration_decay_after=1,
    update_target_after_steps=1_000
)
agent = Agent(interpreter, driver_algorithm)
for i in range(1_0):
    print("Training Iteration: ", i)
    agent.train(initial_episode=100 * i, episodes=100, batch_size=64, metric=metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(
#     learn_after_steps=3,
#     replay_size= 1_00_000,
#     discount_factor= 0.99,
#     exploration=0.08342596622120578,
#     min_exploration=0.01,
#     exploration_decay=1.005,
#     exploration_decay_after=1,
#     update_target_after_steps=1_000
# )
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(5, 1_0):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, batch_size=64, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
