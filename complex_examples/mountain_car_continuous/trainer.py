import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepDPG
from deep_rl.analytics import AvgTotalReward
from mountaincarcont_env_wrappers import MountainCarContinuousEnvironment

env = MountainCarContinuousEnvironment(gym.make("MountainCarContinuous-v0", render_mode = "rgb_array"))
AGENT_PATH = "mountain_car_cont_agent2"

# Actor Network
state_input = Input(shape=(2,))
x = Dense(64, activation="relu")(state_input)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation='tanh', kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
actor_network = Model(inputs=state_input, outputs=output)
actor_network.compile(optimizer=Adam(learning_rate=0.0005))

# Critic Network
state_input = Input(shape=(2,))
action_input = Input(shape=(1,))
x1 = Dense(32, activation='relu')(state_input)
x2 = Dense(32, activation='relu')(action_input)
x = Concatenate()([x1, x2])
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
critic_network = Model(inputs=[state_input, action_input], outputs=output)
critic_network.compile(optimizer=Adam(learning_rate=0.001))

metric = AvgTotalReward(os.path.join(AGENT_PATH, "train_metric"))

driver_algorithm = DeepDPG(
    actor_network,
    critic_network,
    learn_after_steps=3,
    replay_size=1_00_000,
    discount_factor=0.99,
    exploration=1,
    exploration_decay=1,
    tau=0.001
)
agent = Agent(env, driver_algorithm)
# 1_000 episodes
for i in range(1_00):
    print("Training Iteration:", i)
    agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
    agent.save(AGENT_PATH)
env.close()

# Load agent and train
# driver_algorithm = DeepDPG(
#     learn_after_steps=1,
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
