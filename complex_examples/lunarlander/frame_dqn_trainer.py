import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam

from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import AvgTotalReward
from lunarlander_env_wrappers import LunarLanderImageEnvironment

AGENT_PATH = "lunar_lander_agent3"
FRAME_BUFFER_SIZE = 5
PREPROCESSED_FRAME_SHAPE = (80, 80, 3)
env = LunarLanderImageEnvironment(gym.make("LunarLander-v2", render_mode = "rgb_array"),
                                          frame_buffer_size=FRAME_BUFFER_SIZE,
                                          preprocessed_frame_shape=PREPROCESSED_FRAME_SHAPE)

# Q Network
input = Input(shape=(FRAME_BUFFER_SIZE, 80, 80, 3))
x = Conv2D(filters=16, kernel_size=5, strides=2, activation="relu")(input)
x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = GlobalAveragePooling3D()(x)
x = Dense(16, activation="relu")(x)
x = Dense(16, activation="relu")(x)
output = Dense(4, activation="linear")(x)
q_network = Model(inputs=input, outputs=output)

optimizer = Adam(learning_rate=0.001)
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

agent = Agent(env, driver_algorithm)
for i in range(1_00):
    print("Training Iteration:", i)
    agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
    agent.save(AGENT_PATH)
env.close()

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
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# for i in range(60, 2_00):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, metric=metric, batch_size=64)
#     agent.save(AGENT_PATH)
# env.close()
