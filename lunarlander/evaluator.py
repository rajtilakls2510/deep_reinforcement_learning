import gym, imageio, os
import cv2
from interface import LunarLanderTerminal, LunarLanderInterpreter
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning

interpreter = LunarLanderInterpreter(LunarLanderTerminal(gym.make("LunarLander-v2")))

AGENT_PATH = "lunar_lander_agent"

driver_algo = DeepQLearning()
agent = Agent(interpreter, driver_algo)
agent.load(AGENT_PATH)

# Live agent play
for i in range(5):
    rgb_array = agent.evaluate(episodes=1, exploration = 0.0)[0]
    print("Episode:",i+1, "Length:", len(rgb_array))
    for frame in rgb_array:
        cv2.imshow("Episode", frame)
        cv2.waitKey(10)
interpreter.close()
cv2.destroyAllWindows()

# Store in video
# try:
#     os.makedirs(os.path.join(AGENT_PATH, "eval"))
# except:
#     pass
# rgb_array = agent.evaluate(episodes=2)
# for i, episode in enumerate(rgb_array):
#     with imageio.get_writer(os.path.join(AGENT_PATH, "eval", "vid200_" + str(i) + "_" + str(len(episode)) + ".mp4"),
#                             fps=50) as video:
#         for frame in episode:
#             video.append_data(frame)
