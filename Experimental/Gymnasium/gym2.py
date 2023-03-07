import gymnasium as gym

import sys
import traceback

path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"

try:
    env = gym.make("Ant-v4", render_mode="human", terminate_when_unhealthy=True)

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
except:
    _, _, tb = sys.exc_info()
    traceback.print_tb(tb)
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]

    print('Line {} in statement {}'.format(line, text))



