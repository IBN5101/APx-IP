from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DDPG

from gym_blobby.envs.env_blobby import BlobbyEnv
import gymnasium

# Main paths
xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/output/blobby_"
# Surely there is a better way to do this
total_timesteps = 1 * 1000000
episodes = 10
# --------------------------------
part = 10
# --------------------------------
steps_id = round(total_timesteps / episodes * part)
sb_path += str(steps_id) + "_steps"
# Special paths
# 01: Breakdance (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/breakdance"
# 02: Posing (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/posing"
# 03: Fallback - PPO v0
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/PPO_fallback_v0"
# 04: DDPG test
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/test_ddpg"
# 05: Fallback - DDPG v0
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/DDPG_fallback_v0"

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
    # print(observation[:3].round(3))
    print(str(info["food_reward"]) + "\t" + str(info["HP"]) + "\t" + str(round(reward, 4)))
    # print(str(observation[-3:].round(3)) + " \t " + str(info["HP"]) + " \t " + str(round(reward, 3)))

    
    if terminated or truncated:
        observation, info = env.reset()
env.close()