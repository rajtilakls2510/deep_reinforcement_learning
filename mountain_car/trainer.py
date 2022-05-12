import gym,os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from mountain_car.interface import MountainCarInterpreter, MountainCarTerminal

interpreter = MountainCarInterpreter(MountainCarTerminal(gym.make("MountainCar-v0")))
AGENT_PATH = "mountain_car_agent2"
# Q Network
input = Input(shape=(2,))
output = Dense(3, activation='linear')(input)
q_network = Model(inputs=input, outputs=output)
optimizer = Adam()

metric = AvgTotalReward(os.path.join(AGENT_PATH,"train_metric"))

driver_algorithm = DeepQLearning(q_network, optimizer, exploration=1,exploration_decay_after=100, update_target_after_steps=500, replay_size=10000)
agent = Agent(interpreter, driver_algorithm)
# 10_000 episodes
for i in range(10_0):
    print("Training Iteration:",i)
    agent.train(initial_episode=100 * i, episodes = 100, metric= metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(q_net, optimizer, exploration=0.05, min_exploration=0.01, exploration_decay=1.15,
#                                  exploration_decay_after=100)
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(10_0, 20_0):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
