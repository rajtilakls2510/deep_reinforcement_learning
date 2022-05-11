import gym
import cv2
from interface import CartpoleTerminal, CartpoleInterpreter
from deep_rl.agent import Agent
from deep_rl.algorithms import DeepQLearning

env = gym.make("CartPole-v1")
interpreter = CartpoleInterpreter(CartpoleTerminal(env))

AGENT_PATH = "cart_pole_agent2"

driver_algo = DeepQLearning()
agent = Agent(interpreter, driver_algo)
agent.load(AGENT_PATH)

for i in range(5):
    rgb_array = agent.infer(episodes=1, exploration = 0.0)[0]
    print("Episode:",i+1, "Length:", len(rgb_array))
    for frame in rgb_array:
        cv2.imshow("Episode", frame)
        cv2.waitKey(20)
interpreter.close()
cv2.destroyAllWindows()