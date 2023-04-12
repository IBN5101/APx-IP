from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import TD3

from env_blobby import BlobbyEnv
import gymnasium

# Path - Output
xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/output/blobby_"
# (IBN) Surely there is a better way to do this
total_timesteps = 10 * 1000000
episodes = 10
# --------------------------------
part = 9
# --------------------------------
steps_id = round(total_timesteps / episodes * part)
sb_path += str(steps_id) + "_steps"
# Path - Specials
# 01: Breakdance (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/breakdance"
# 02: Posing (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/posing"
# 03: Fallback - PPO v0 (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/PPO_fallback_v0"
# 04: DDPG test (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/test_ddpg"
# 05: Fallback - DDPG v0 (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/DDPG_fallback_v0"
# 06: Testing - A2C (legacy)
# sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/A2C_testing"
# 07: Fallback - PPO v1 
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/PPO_fallback_v1"

print("Loading model from: " + sb_path.split("/")[-1] + " ...")

# (IBN) Check if this matches sb3_save.py settings (except render_mode)
env = BlobbyEnv(render_mode="human", xml_file=xml_path)
check_env(env)
# SB3
model = PPO.load(sb_path)
# model = DDPG.load(sb_path)
# model = A2C.load(sb_path)
# model = SAC.load(sb_path)
# model = TD3.load(sb_path)

observation, info = env.reset()
for _ in range(10000):
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    # Debug:
    print(str(info["food_eaten_total"]) + "\t" 
          + str(info["HP"]) + "\t" 
          + str(round(info["closest_food_distance"], 4)) + "\t")
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()