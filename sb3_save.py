import os

from env_blobby import BlobbyEnv
import gymnasium

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import TD3

# Paths
cwd = os.getcwd()
xml_path = os.path.join(cwd, "blobby.xml")
sb_path = "output/"
log_path = "logs/"
monitor_path = "logs/"
# Retrain (PPO)
# UPDATE PER RE-TRAINING
retrain_path = "output/PPO_retrain_v2.2"

# Env setup
env = BlobbyEnv(render_mode=None, xml_file=xml_path)
env = Monitor(env, filename=monitor_path, info_keywords=("food_eaten_total","penalty",))
check_env(env)

# SB3 algorithms
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
# model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
# model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
# model = TD3("MlpPolicy", env, verbose=0, tensorboard_log=log_path)

# SB3 retrain
# model = PPO.load(retrain_path, env=env)

# Training settings
total_timesteps = 15 * 1000000
episodes = 15
# Estimations:
# > PPO
#   1M steps = 41 mins
#   3M steps = 2 hours
#   5M steps = 3 hours
#  10M steps = 6 hours
#  15M steps = 11 hours
# > DDPG
# 300k steps = 3 hours
#   1M steps = 6 hours
#   2M steps = 12 hours
# > A2C
#   1M steps = 50 mins
#  10M steps = 8 hours
# > SAC
#   1M steps = 10 hours
# > TD3
#   1M steps = 7 hours

episode_timesteps = total_timesteps / episodes
checkpoint_callback = CheckpointCallback(
    save_freq=episode_timesteps,
    save_path=sb_path,
    name_prefix="blobby",
)

model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=checkpoint_callback)
print("[!] Training complete.")