import gym,os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from mountain_car.interface import MountainCarInterpreter, MountainCarTerminal

gym.envs.registration.register(
    id="MountainCar1000-v0",
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1000,  # MountainCar-v0 uses 200,
    reward_threshold=-110.0,
)

interpreter = MountainCarInterpreter(MountainCarTerminal(gym.make("MountainCar1000-v0")))
AGENT_PATH = "mountain_car_agent"
# Q Network
input = Input(shape=(2,))
x = Dense(100, activation="sigmoid", kernel_initializer="zeros")(input)
output = Dense(3, activation='linear', kernel_initializer="zeros")(x)
q_network = Model(inputs=input, outputs=output)
optimizer = Adam(learning_rate=0.001)
q_network.compile(optimizer=optimizer)
metric = AvgTotalReward(os.path.join(AGENT_PATH,"train_metric"))

driver_algorithm = DeepQLearning(q_network,  exploration=0.6, min_exploration=0.0, exploration_decay_after=50, replay_size=10_000)
agent = Agent(interpreter, driver_algorithm)
# 10_000 episodes
for i in range(10_00):
    print("Training Iteration:",i)
    agent.train(initial_episode=10 * i, episodes = 10, metric= metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(exploration=0.07, min_exploration=0.0, exploration_decay=1.1,
#                                  exploration_decay_after=50, replay_size=10_000)
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(50, 10_00):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
