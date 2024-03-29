import os
import gym
from cartpole_env_wrappers import CartpoleEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, ExplorationTrackerMetric, AbsoluteValueErrorMetric, SaveAgent
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# from tensorflow import config
#
# physical_devices = config.list_physical_devices('GPU')
# try:
#   config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

# Wrapping the gym environment to interface with our library
env = CartpoleEnvironment(gym.make("CartPole-v1", render_mode="rgb_array"))

# Path to Agent folder
AGENT_PATH = "cart_pole_agent2"

# Q Network
net_input = Input(shape=(4,))
x = Dense(32, activation="relu")(net_input)
x = Dense(16, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)
optimizer = Adam()
q_net.compile(optimizer=optimizer)

# Setting up metrics for training
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
exp_tracker = ExplorationTrackerMetric(os.path.join(AGENT_PATH, "train_metric"))
value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))
save_agent = SaveAgent("cartpole_checkpoints", save_after_episodes=50)

# =============== Starting training from scratch ======================

# Creating the algorithm that will be used to train the agent
driver_algorithm = DeepQLearning(
    q_net,
    learn_after_steps=3,
    replay_size=1_00_000,
    discount_factor=0.99,
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1.01,
    exploration_decay_after=1,
    update_target_after_steps=1_000
)
# Creating the Agent class
agent = Agent(env, driver_algorithm)

# Training for 500 episodes. Saving the agent every 100 episodes
for i in range(5):
    print("Training Iteration: ", i)
    agent.train(
        initial_episode=100 * i,
        episodes=100,
        batch_size=64,
        metrics=[ep_length, total_reward, avg_q, exp_tracker, value_error, save_agent]
    )
    agent.save(AGENT_PATH)
env.close()

# ============================== Loading the agent and resuming training ================
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
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# # Resuming training from 200th episode and training till 1_000th episode
# for i in range(2, 1_0):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, batch_size=64, metrics=[ep_length, total_reward, avg_q, exp_tracker, value_error])
#     agent.save(AGENT_PATH)
# env.close()
