import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from car_racing.interface import CarRacingTerminal, CarRacingInterpreter

interpreter = CarRacingInterpreter(CarRacingTerminal(gym.make("CarRacingDiscrete-v1")))
AGENT_PATH = "car_racing_agent"

# Q Network
input = Input(shape=(96, 96, 3 * 3))
x = Conv2D(32, 3, activation='relu')(input)
x = BatchNormalization()(x)
x = Conv2D(16, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation="relu")(x)
output = Dense(5, activation="linear")(x)
q_network = Model(inputs=input, outputs=output)

optimizer = Adam(learning_rate=0.0005)
q_network.compile(optimizer=optimizer)

metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"))

# Start training from scratch
driver_algorithm = DeepQLearning(
    q_network,
    learn_after_steps=4,
    discount_factor=0.99,
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1.005,
    exploration_decay_after=1,
    update_target_after_steps=1_000,
    replay_size=1_00_000
)

agent = Agent(interpreter, driver_algorithm)
agent.train(episodes=1, metric=metric, batch_size=64)
agent.save(AGENT_PATH)
# for i in range(1_00):
#     print("Training Iteration:", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
#     agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(
#     learn_after_steps=4,
#     discount_factor=0.999,
#     exploration=0.20371654390816973,
#     min_exploration=0.01,
#     exploration_decay=1.005,
#     exploration_decay_after=1,
#     update_target_after_steps=1_000,
#     replay_size=1_00_000
# )
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(60, 2_00):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
#     agent.save(AGENT_PATH)
# interpreter.close()
