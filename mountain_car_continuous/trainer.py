import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import AvgTotalReward
from mountain_car_continuous.interface import MountainCarContinuousInterpreter, MountainCarContinuousTerminal

interpreter = MountainCarContinuousInterpreter(MountainCarContinuousTerminal(gym.make("MountainCarContinuous-v0")))
AGENT_PATH = "mountain_car_cont_agent"

# Actor Network
state_input = Input(shape=(2,))
x = Dense(32, activation="relu")(state_input)
output = Dense(1, activation='tanh')(x)
actor_network = Model(inputs=state_input, outputs=output)
actor_network.compile(optimizer=Adam(learning_rate=0.0001))

# Critic Network
state_input = Input(shape=(2,))
action_input = Input(shape=(1,))
x1 = Dense(32, activation='relu')(state_input)
x2 = Dense(32, activation='relu')(action_input)
x = Add()([x1, x2])
output = Dense(1, activation="linear")(x)
critic_network = Model(inputs=[state_input, action_input], outputs=output)
critic_network.compile(optimizer=Adam(learning_rate=0.001))

metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"), continuous=True)

driver_algorithm = DeepDPG(
    actor_network,
    critic_network,
    learn_after_steps=4,
    replay_size=1_00_000,
    discount_factor=0.99,
    tau=0.001
)
agent = Agent(interpreter, driver_algorithm)
# 1_000 episodes
for i in range(1_00):
    print("Training Iteration:", i)
    agent.train(initial_episode=10 * i, episodes=10, metric=metric)
    agent.save(AGENT_PATH)
interpreter.close()

# Load agent and train
# driver_algorithm = DeepDPG(
#     learn_after_steps=4,
#     replay_size=1_00_000,
#     discount_factor=0.99,
#     tau=0.001
# )
#
# agent = Agent(interpreter, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(1_00, 1_50):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric)
#     agent.save(AGENT_PATH)
# interpreter.close()
