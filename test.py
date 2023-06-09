import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from pettingzoo.mpe import simple_adversary_v3
from deep_rl.magent_parallel import MAParallelEnvironment, MADDPG, MATotalRewardMetric

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

env = MAParallelEnvironment(simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True, render_mode="rgb_array"))

# Actor network
obs_input = layers.Input(shape=(10,))
x = layers.Dense(16, activation="relu")(obs_input)
out = layers.Dense(5, activation="sigmoid")(x)
actor_network = Model(inputs=obs_input, outputs=out)
actor_network.compile(optimizer=optimizers.Adam(learning_rate=0.001))

# Adversary Actor network
obs_input = layers.Input(shape=(8,))
x = layers.Dense(16, activation="relu")(obs_input)
out = layers.Dense(5, activation="sigmoid")(x)
actor_adversary_network = Model(inputs=obs_input, outputs=out)
actor_adversary_network.compile(optimizer=optimizers.Adam(learning_rate=0.001))

# Critic Network
obs_inp = layers.Input(shape=(2,10))
x1 = layers.Dense(16, activation="relu")(obs_inp)
x1 = layers.Flatten()(x1)

act_inp = layers.Input(shape=(2,5))
x2 = layers.Dense(16, activation="relu")(act_inp)
x2 = layers.Flatten()(x2)

x = layers.Add()([x1,x2])
out = layers.Dense(1)(x)
critic_network = Model(inputs=(obs_inp, act_inp), outputs=out)
critic_network.compile(optimizer=optimizers.Adam(learning_rate=0.001))

# Adversary Critic Network
obs_inp = layers.Input(shape=(1,8))
x1 = layers.Dense(16, activation="relu")(obs_inp)
x1 = layers.Flatten()(x1)

act_inp = layers.Input(shape=(1,5))
x2 = layers.Dense(16, activation="relu")(act_inp)
x2 = layers.Flatten()(x2)

x = layers.Add()([x1, x2])
out = layers.Dense(1)(x)
critic_adversary_network = Model(inputs=(obs_inp, act_inp), outputs=out)
critic_adversary_network.compile(optimizer=optimizers.Adam(learning_rate=0.001))

total_reward = MATotalRewardMetric(path="agent")

algorithm = MADDPG(env, env.agents, env.adversaries, actor_network, actor_adversary_network,
                   critic_network, critic_adversary_network, replay_size=10000, exploration=0.5)
algorithm.train(initial_episode=0, episodes=100, batch_size=16, metrics=[total_reward])





