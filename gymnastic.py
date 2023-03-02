from gym_blobby.envs.env_blobby import BlobbyEnv

xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"

env = BlobbyEnv(render_mode="human", xml_file=xml_path, terminate_when_unhealthy=True)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()