from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import TD3

from gym_blobby.envs.env_blobby import BlobbyEnv
import gymnasium

# Settings
xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/output/"
tb_log = "/home/vboxuser/Desktop/HQplus/CS3IP/tb_log/"

# Notes
# Idea 1: Change food to not be one-time reward
# Idea 2: Exponential reward to distance to food
# Idea 3: Termination with health. Health = 100, decrease on time, increase on food.
# Idea 4: Death = -1, Food = +1, HP / 10000?

def sb3_save():
    env = BlobbyEnv(render_mode=None, xml_file=xml_path)

    check_env(env)

    # SB3 algorithms
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tb_log)
    # model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=tb_log)
    # model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=tb_log)
    # model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=tb_log)
    # model = TD3("MlpPolicy", env, verbose=0, tensorboard_log=tb_log)
    # Estimations:
    # <!> PPO
    #   1M steps = 41 mins
    #   3M steps = 2 hours
    #   5M steps = 3 hours
    #  10M steps = 6 hours
    # <!> DDPG
    # 300k steps = 3 hours
    #   1M steps = 6 hours
    # <!> A2C
    #   1M steps = 50 mins
    #  10M steps = 8 hours
    # <!> SAC
    #   1M steps = 10 hours
    # <!> TD3
    #
    total_timesteps = 0.1 * 1000000
    episodes = 10

    episode_timesteps = total_timesteps / episodes
    checkpoint_callback = CheckpointCallback(
        save_freq=episode_timesteps,
        save_path=sb_path,
        name_prefix="blobby",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=checkpoint_callback)
    # for i in range(episodes):
    #     model.learn(total_timesteps=episode_timesteps, progress_bar=True, reset_num_timesteps=False)
    #     model.save(f"{sb_path}{i}")
    #     print("[-] Episode " + str(i) + " complete.")
    
    print("[!] Training complete.")

def sb3_load():
    env = BlobbyEnv(render_mode="human", xml_file=xml_path)
    model = PPO.load(sb_path)

    observation, info = env.reset()
    for _ in range(1000):
        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()


# Main
sb3_save()
# sb3_load()