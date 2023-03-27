from gym_blobby.envs.env_blobby import BlobbyEnv

xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"

env = BlobbyEnv(render_mode="human", xml_file=xml_path, terminate_when_unhealthy=False)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(info["food_distances"].round(2))
    # print(info["closest_food_distance"].round(3))
    print(str(info["food_eaten_total"]) + "\t" + str(info["HP"]) + "\t" + str(round(reward, 6)))
    # print(info["sensor_data"])
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()