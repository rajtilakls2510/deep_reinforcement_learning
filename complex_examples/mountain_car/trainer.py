import gym,os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
from deep_rl.analytics import EpisodeLengthMetric, TotalRewardMetric, AverageQMetric, ExplorationTrackerMetric, AbsoluteValueErrorMetric
from mountaincar_env_wrappers import MountainCarEnvironment

# Wrapping the gym environment to interface with our library
env = MountainCarEnvironment(gym.make("MountainCar-v0", render_mode = "rgb_array"))
AGENT_PATH = "mountain_car_agent"

# Q Network
input = Input(shape=(2,))
x = Dense(32, activation="relu")(input)
output = Dense(3, activation='linear')(x)
q_network = Model(inputs=input, outputs=output)
optimizer = Adam(learning_rate=0.001)
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
    exploration=1,
    min_exploration=0.01,
    exploration_decay=1.05,
    exploration_decay_after=1,
    update_target_after_steps=1_000
)
# Creating the Agent class
agent = Agent(env, driver_algorithm)

# Training for 300 episodes. Saving the agent every 10 episodes
for i in range(30):
    print("Training Iteration:",i)
    agent.train(
        initial_episode=10 * i,
        episodes=10,
        batch_size=64,
        metrics=[ep_length, total_reward, avg_q, exp_tracker, value_error]
    )
    agent.save(AGENT_PATH)
env.close()

# Load agent and train (change exploration param)
# driver_algorithm = DeepQLearning(
#     q_network,
#     learn_after_steps=4,
#     replay_size=1_00_000,
#     discount_factor=0.99,
#     exploration=0.01,
#     min_exploration=0.01,
#     exploration_decay=1.05,
#     exploration_decay_after=1,
#     update_target_after_steps=1_000
# )
#
# agent = Agent(env, driver_algorithm)
# agent.load(AGENT_PATH)
# # Resuming training from 300th episode and training till 400th episode
# for i in range(30, 40):
#     print("Training Iteration: ", i)
#     agent.train(initial_episode=10 * i, episodes=10, batch_size=64, metrics=[ep_length, total_reward, avg_q, exp_tracker, value_error])
#     agent.save(AGENT_PATH)
# env.close()
