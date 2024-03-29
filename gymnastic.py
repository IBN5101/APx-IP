import os
from env_blobby import BlobbyEnv

cwd = os.getcwd()
xml_path = os.path.join(cwd, "blobby.xml")

env = BlobbyEnv(render_mode="human", xml_file=xml_path)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(info["food_distances"].round(2))
    # print(info["closest_food_distance"].round(3))
    # print(str(info["food_eaten_total"]) + "\t" + str(info["HP"]) + "\t" + str(round(reward, 6)))
    # print(str(round(info["penalty"])) + "\t" + str(round(observation[-4], 5)))
    # print(observation[-2] + "\t" + str(round(observation[-3], 5)))
    # print(observation[-1])
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()