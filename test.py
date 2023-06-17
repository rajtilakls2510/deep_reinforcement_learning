import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from pettingzoo.mpe import simple_adversary_v3
from deep_rl.analytics import LiveEpisodeViewer, EpisodeLengthMetric
import os
from deep_rl.magent_parallel import MAParallelEnvironment, MADDPG, MATotalRewardMetric

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

env = MAParallelEnvironment(simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode="rgb_array"))

AGENT_PATH = "agent"

# Actor network
obs_input = layers.Input(shape=(10,))
x = layers.Dense(128, activation="relu")(obs_input)
out = layers.Dense(5, activation="sigmoid")(x)
actor_network = Model(inputs=obs_input, outputs=out)
actor_network.compile(optimizer=optimizers.Adam(learning_rate=0.0001))

# Adversary Actor network
obs_input = layers.Input(shape=(8,))
x = layers.Dense(128, activation="relu")(obs_input)
out = layers.Dense(5, activation="sigmoid")(x)
actor_adversary_network = Model(inputs=obs_input, outputs=out)
actor_adversary_network.compile(optimizer=optimizers.Adam(learning_rate=0.0001))

# Critic Network
# Good Agents Heads
obs_agent_inp = layers.Input(shape=(2,10))
x1 = layers.Dense(96, activation="relu")(obs_agent_inp)
x1 = layers.Flatten()(x1)

act_agent_inp = layers.Input(shape=(2,5))
x2 = layers.Dense(96, activation="relu")(act_agent_inp)
x2 = layers.Flatten()(x2)

# Adversary Agents Heads
obs_adversary_inp = layers.Input(shape=(1,8))
x3 = layers.Dense(192, activation="relu")(obs_adversary_inp)
x3 = layers.Flatten()(x3)

act_adversary_inp = layers.Input(shape=(1,5))
x4 = layers.Dense(192, activation="relu")(act_adversary_inp)
x4 = layers.Flatten()(x4)

# Fusing All Heads
x = layers.Add()([x1,x2, x3, x4])
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(1)(x)
critic_network = Model(inputs=(obs_agent_inp, act_agent_inp, obs_adversary_inp, act_adversary_inp), outputs=out)
critic_network.compile(optimizer=optimizers.Adam(learning_rate=0.001))

total_reward = MATotalRewardMetric(path=os.path.join(AGENT_PATH, "train_metric"))
ep_length = EpisodeLengthMetric(path=os.path.join(AGENT_PATH, "train_metric"))
live = LiveEpisodeViewer(fps=600)

algorithm = MADDPG(env, env.agents, env.adversaries, actor_network, actor_adversary_network,
                   critic_network, replay_size=100000, exploration=1.0)
algorithm.train(initial_episode=0, episodes=25000, batch_size=32, metrics=[total_reward, ep_length, live])
algorithm.save(AGENT_PATH)




