import gym, imageio, os
import cv2
from mountaincar_env_wrappers import MountainCarShapedEnvironment
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning
import numpy as np

env = MountainCarShapedEnvironment(gym.make("MountainCar-v0", render_mode = "rgb_array"))

AGENT_PATH = "mountain_car_shaped_agent3"

driver_algo = DeepQLearning()
agent = Agent(env, driver_algo)
agent.load(AGENT_PATH)

# Live agent play
def put_episodic_data(episode):
    colour = (255,0,0)
    episode[0] = cv2.putText(episode[0], "Step: " + str(step), org=(10, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.3, color=colour, lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "State: " + str(np.round(episode[2], decimals=3)), org=(10, 24),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=colour,
                             lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "Action: " + actions[episode[3]], org=(10, 36),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=colour,
                             lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "Value: " + str(episode[4]), org=(10, 48),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=colour,
                             lineType=cv2.LINE_AA)
    episode[0] = cv2.putText(episode[0], "Reward: " + str(episode[1]), org=(10, 60),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=colour,
                             lineType=cv2.LINE_AA)
    # return episode


actions = ['L', 'D', 'R']
for i in range(5):
    rgb_array = agent.evaluate(episodes=1, exploration=0.0)[0]
    print("Episode:", i + 1, "Length:", len(rgb_array))
    tot_reward = 0
    for step, episode in enumerate(rgb_array):
        put_episodic_data(episode)
        tot_reward += episode[1]
        cv2.imshow("Episode", episode[0])
        cv2.waitKey(20)
    print("Total Reward:", tot_reward)
env.close()
cv2.destroyAllWindows()

# Store in video
# try:
#     os.makedirs(os.path.join(AGENT_PATH, "eval"))
# except:
#     pass
# rgb_array = agent.evaluate(episodes=5)
# for i, episode in enumerate(rgb_array):
#     with imageio.get_writer(os.path.join(AGENT_PATH, "eval", "vid" + str(i) + "_" + str(len(episode)) + ".mp4"),
#                             fps=30) as video:
#         for ep in episode:
#             video.append_data(ep[0])
