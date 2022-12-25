import gym, os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from deep_rl.agent import Agent, GymEnvironment
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, ExplorationTrackerMetric, AbsoluteValueErrorMetric


# Wrapping the gym environment to interface with our library
# (Did not create any custom wrapper. Just using the default wrapper for gym environments from this library)
env = GymEnvironment(gym.make("LunarLander-v2", render_mode = "rgb_array"))

# Path to Agent folder
AGENT_PATH = "lunar_lander_agent"

# Q Network
input = Input(shape=(8,))
x = Dense(64, activation="relu")(input)
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
output = Dense(4, activation="linear")(x)
q_network = Model(inputs=input, outputs=output)
optimizer = Adam(learning_rate=0.0005)
q_network.compile(optimizer=optimizer)


# Setting up metrics for training
ep_length = EpisodeLengthMetric(os.path.join(AGENT_PATH, "train_metric"))
total_reward = TotalRewardMetric(os.path.join(AGENT_PATH, "train_metric"))
avg_q = AverageQMetric(os.path.join(AGENT_PATH, "train_metric"))
exp_tracker = ExplorationTrackerMetric(os.path.join(AGENT_PATH, "train_metric"))
value_error = AbsoluteValueErrorMetric(os.path.join(AGENT_PATH, "train_metric"))

# =============== Starting training from scratch ======================

# Creating the algorithm that will be used to train the agent
driver_algorithm = DeepQLearning(
    q_network,
    learn_after_steps=4,
    replay_size=1_00_000,
    discount_factor=0.99,
    exploration=1.0,
    min_exploration=0.01,
    exploration_decay=1.005,
    exploration_decay_after=1,
    update_target_after_steps=1_000
)
# Creating the Agent class
agent = Agent(env, driver_algorithm)

# Training for 2_500 episodes. Saving the agent every 100 episodes
for i in range(2_5):
    print("Training Iteration: ", i)
    agent.train(
        initial_episode=100 * i,
        episodes=100,
        batch_size=64,
        metrics=[ep_length, total_reward, avg_q, exp_tracker, value_error]
    )
    agent.save(AGENT_PATH)
env.close()

# ============================== Loading the agent and resuming training ================
# driver_algorithm = DeepQLearning(
#     learn_after_steps=4,
#     replay_size= 1_00_000,
#     discount_factor= 0.99,
#     exploration=0.0,
#     min_exploration=0.01,
#     exploration_decay=1.005,
#     exploration_decay_after=1,
#     update_target_after_steps=1_000
# )
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# # Resuming training from 800th episode and training till 1_200th episode
# for i in range(8, 1_2):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=100 * i, episodes=100, batch_size=64, metrics=[ep_length, total_reward, avg_q, exp_tracker, value_error])
#     agent.save(AGENT_PATH)
# env.close()
