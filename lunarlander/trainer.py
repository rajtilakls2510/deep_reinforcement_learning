import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from lunarlander.interface import LunarLanderTerminal, LunarLanderInterpreter

interpreter = LunarLanderInterpreter(LunarLanderTerminal(gym.make("LunarLander-v2")))
AGENT_PATH = "lunar_lander_agent2"

# Q Network
input = Input(shape=(8,))
x = Dense(256, activation="sigmoid")(input)
x = Dense(64, activation="relu")(x)
output = Dense(4, activation="linear")(x)
q_network = Model(inputs=input, outputs=output)

optimizer = Adam()

metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"))

# Start training from scratch
driver_algorithm = DeepQLearning(
    q_network,
    optimizer,
    exploration=1,
    min_exploration=0.0,
    exploration_decay=1.1,
    exploration_decay_after=50,
    update_target_after_steps=100,
    replay_size=10_000
)

agent = Agent(interpreter, driver_algorithm)
for i in range(10_00):
    print("Training Iteration:", i)
    agent.train(initial_episode=10*i, episodes=10, metric=metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(
#     optimizer=optimizer,
#     exploration=0.1,
#     min_exploration=0.0,
#     exploration_decay=1.2,
#     exploration_decay_after=100,
#     update_target_after_steps=100,
#     replay_size=10_000
# )
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(1_9, 10_0):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
