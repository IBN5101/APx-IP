from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DDPG

from gym_blobby.envs.env_blobby import BlobbyEnv
import gymnasium

# Main paths
xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/result"
# Special paths
# 01: Breakdance (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/breakdance"
# 02: Posing (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/posing"
# 03: Fallback (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/fallback_v0"
# 04: DDPG test
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/test_ddpg"

# Check if this matches sb3_save.py settings
env = BlobbyEnv(render_mode="human", xml_file=xml_path)
# env = gymnasium.make("Ant-v4", render_mode = "human", terminate_when_unhealthy=True)

# SB3
check_env(env)
# model = PPO.load(sb_path)
model = DDPG.load(sb_path)

observation, info = env.reset()
for _ in range(10000):
    # action = env.action_space.sample()
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    # print(info["closest_food_distance"])
    print(reward)
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()