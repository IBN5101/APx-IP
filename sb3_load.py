from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym_blobby.envs.env_blobby import BlobbyEnv
import gymnasium

xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/ppo_blobby"

# Check if this matches sb3_save.py settings
env = BlobbyEnv(render_mode="human", xml_file=xml_path, terminate_when_unhealthy=True)
# env = gymnasium.make("Ant-v4", render_mode = "human", terminate_when_unhealthy=True)

# SB3
check_env(env)
model = PPO.load(sb_path)

observation, info = env.reset()
for _ in range(1000):
    # action = env.action_space.sample()
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()