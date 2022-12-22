import os
import gym
from cartpole_env_wrappers import CartpoleEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, ExplorationTrackerMetric
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

env = CartpoleEnvironment(gym.make("CartPole-v1", render_mode = "rgb_array"))

AGENT_PATH = "cart_pole_agent3"

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

# Start training from scratch
driver_algorithm = DeepQLearning(
    q_net,
    learn_after_steps=3,
    replay_size= 1_00_000,
    discount_factor= 0.99,
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1.1,
    exploration_decay_after=1,
    update_target_after_steps=1_000
)
agent = Agent(env, driver_algorithm)
for i in range(1):
    print("Training Iteration: ", i)
    agent.train(
        initial_episode=100 * i,
        episodes=100,
        batch_size=64,
        metrics=[ep_length, total_reward, avg_q, exp_tracker]
    )
    agent.save(AGENT_PATH)
env.close()

# Load agent and train (change exploration param)
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
# for i in range(2, 1_0):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, batch_size=64, metrics=[ep_length, total_reward, avg_q, exp_tracker])
#     agent.save(AGENT_PATH)
# env.close()
