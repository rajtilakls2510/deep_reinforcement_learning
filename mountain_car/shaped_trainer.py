import gym,os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from mountain_car.interface import MountainCarShapedInterpreter, MountainCarTerminal

interpreter = MountainCarShapedInterpreter(MountainCarTerminal(gym.make("MountainCar-v0")))
AGENT_PATH = "mountain_car_shaped_agent3"
# Q Network
input = Input(shape=(2,))
x = Dense(32, activation="relu")(input)
output = Dense(3, activation='linear')(x)
q_network = Model(inputs=input, outputs=output)
optimizer = Adam(learning_rate=0.001)
q_network.compile(optimizer=optimizer)

metric = AvgTotalReward(os.path.join(AGENT_PATH,"train_metric"))

driver_algorithm = DeepQLearning(
    q_network,
    learn_after_steps=4,
    replay_size=1_00_000,
    discount_factor=0.99,
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1.01,
    exploration_decay_after=1,
    update_target_after_steps=1_000
)
agent = Agent(interpreter, driver_algorithm)
# 1_000 episodes
for i in range(1_00):
    print("Training Iteration:",i)
    agent.train(initial_episode=10 * i, episodes = 10, metric= metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(
#     learn_after_steps=4,
#     replay_size=1_00_000,
#     discount_factor=0.99,
#     exploration=0.01,
#     min_exploration=0.01,
#     exploration_decay=1.005,
#     exploration_decay_after=1,
#     update_target_after_steps=1_000
# )
#
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(1_00, 1_20):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
