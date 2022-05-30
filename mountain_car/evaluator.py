import gym, imageio, os
import cv2
from interface import MountainCarInterpreter, MountainCarTerminal
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning

gym.envs.registration.register(
    id="MountainCar1000-v0",
    entry_point='gym.envs.classic_control.mountain_car:MountainCarEnv',
    max_episode_steps=1000,  # MountainCar-v0 uses 200,
    reward_threshold=-110.0,
)

interpreter = MountainCarInterpreter(MountainCarTerminal(gym.make("MountainCar1000-v0")))

AGENT_PATH = "mountain_car_agent"

driver_algo = DeepQLearning()
agent = Agent(interpreter, driver_algo)
agent.load(AGENT_PATH)

# Live agent play
# for i in range(5):
#     rgb_array = agent.evaluate(episodes=1, exploration = 0.0)[0]
#     print("Episode:",i+1, "Length:", len(rgb_array))
#     for episode in rgb_array:
#         cv2.imshow("Episode", episode[0])
#         cv2.waitKey(10)
# interpreter.close()
# cv2.destroyAllWindows()

# Store in video
try:
    os.makedirs(os.path.join(AGENT_PATH, "eval"))
except:
    pass
rgb_array = agent.evaluate(episodes=5)
for i, episode in enumerate(rgb_array):
    with imageio.get_writer(os.path.join(AGENT_PATH, "eval", "vid" + str(i) + "_" + str(len(episode)) + ".mp4"),
                            fps=30) as video:
        for ep in episode:
            video.append_data(ep[0])
