import gym, imageio, os
import cv2
from interface import LunarLanderTerminal, LunarLanderInterpreter
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning

interpreter = LunarLanderInterpreter(LunarLanderTerminal(gym.make("LunarLander-v2")))

AGENT_PATH = "lunar_lander_agent2"

driver_algo = DeepQLearning()
agent = Agent(interpreter, driver_algo)
agent.load(AGENT_PATH)


# Live agent play

def put_episodic_data(episode):
    episode[0] = cv2.putText(episode[0], "Step: " + str(step), org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.3, color=(0, 255, 0), lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "State: " + str(episode[2]), org=(10, 20),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0),
                             lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "Action: " + actions[episode[3]], org=(10, 30),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0),
                             lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "Value: " + str(episode[4]), org=(10, 40),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0),
                             lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "Reward: " + str(episode[1]), org=(10, 50),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0),
                             lineType=cv2.LINE_AA)
    # return episode


actions = ['D', 'L', 'M', 'R']
for i in range(5):
    rgb_array = agent.evaluate(episodes=1, exploration=0.0)[0]
    print("Episode:", i + 1, "Length:", len(rgb_array))
    for step, episode in enumerate(rgb_array):
        put_episodic_data(episode)
        cv2.imshow("Episode", episode[0])
        cv2.waitKey(400)
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
#         for episode in episode:
#             video.append_data(episode[0])
