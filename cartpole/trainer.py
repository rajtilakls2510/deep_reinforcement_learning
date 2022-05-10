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

AGENT_PATH = "cart_pole_agent2"

# Q Network

net_input = Input(shape=(4,))
x = Dense(30, activation="sigmoid")(net_input)
x = Dense(20, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

optimizer = Adam()
metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"))

# # Start training from scratch
driver_algorithm = DeepQLearning(q_net, optimizer, exploration=1, min_exploration=0.1, exploration_decay=1.1,
                                            exploration_decay_after=100)
agent = Agent(interpreter, driver_algorithm)
for i in range(1):
    print("Training Iteration: ", i)
    agent.train(initial_episode=100 * i, episodes=100, metric=metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(q_net, optimizer, exploration=0.05, min_exploration=0.01, exploration_decay=1.15,
#                                  exploration_decay_after=100)
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(1, 2):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
