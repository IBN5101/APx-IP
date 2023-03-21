from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DDPG

from gym_blobby.envs.env_blobby import BlobbyEnv
import gymnasium

# Settings
xml_path = "/home/vboxuser/Desktop/HQplus/CS3IP/blobby.xml"
sb_path = "/home/vboxuser/Desktop/HQplus/CS3IP/model/result"
unhealthy_termination = False

# Notes
# Idea 1: Change food to not be one-time reward
# Idea 2: Exponential reward to distance to food
# Idea 3: Termination with health. Health = 100, decrease on time, increase on food.
# Idea 4: Death = -1, Food = +1, HP / 10000?

def sb3_save():
    env = BlobbyEnv(render_mode=None, xml_file=xml_path)

    check_env(env)

    # SB3 algorithms
    # model = PPO("MlpPolicy", env, verbose=0)
    model = DDPG("MlpPolicy", env, verbose=0)
    # Estimation (PPO): 
    # 100k steps = 4 mins
    # 1M steps = 41 mins
    # 3M steps = 2 hours
    # 5M steps = 3 hours
    # 10M steps = 6 hours
    model.learn(total_timesteps=0.3 * 1000000, progress_bar=True)

    model.save(sb_path)
    print("Training complete")

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