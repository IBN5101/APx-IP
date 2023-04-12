import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class BlobbyEnv(MujocoEnv, utils.EzPickle):
    # This environment is based on the Ant-v4 environment from Gymnasium
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file="blobby.xml",
        reset_noise_scale=0.1,
        initial_HP=200,
        food_HP_increase=100,
        food_progress_threshold=10,
        floor_z_threshold=0.26,
        HP_loss_upon_touching_floor=10,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            reset_noise_scale,
            initial_HP,
            food_HP_increase,
            food_progress_threshold,
            floor_z_threshold,
            HP_loss_upon_touching_floor,
            **kwargs
        )

        # Initialize custom variables
        self._reset_noise_scale = reset_noise_scale
        self._initial_HP = initial_HP
        self._food_HP_increase = food_HP_increase
        self._food_progress_threshold = food_progress_threshold
        self._floor_z_threshold = floor_z_threshold
        self._HP_loss_upon_touching_floor = HP_loss_upon_touching_floor

        # [!!!] HARD CODING BECAUSE THE MODEL IS INITIALIZED AFTER THE OBSERVATION SPACE
        # (IBN) Observation shape:
        # - Joint position:
        #   Free joint = 7
        #   Fixed joint = 1
        # - Joint velocity:
        #   Free joint = 6
        #   Fixed joint = 1
        # > Blobby has 1 free joint and 12 fixed joint
        # obs_shape += (7 + 12 * 1) + (6 + 12 * 1) = 37
        # > Sphere xpos [z] = 1
        # obs_shape += 1 = 38
        # > Closest food = 1
        # obs_shape += 1 = 39
        # > Food eaten count = 1
        # obs_shape += 1 = 40
        # > Current HP = 1
        # obs_shape += 1 = 41
        obs_shape = 41

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        # (IBN) Initalize custom variables
        self.reset_custom_parameters()

    @property
    def terminated(self):
        terminated = False
        # HP-based termination
        if (self.get_HP() <= 0):
            terminated = True
        # Finite state check
        if (not np.isfinite(self.state_vector()).all()):
            terminated = True

        return terminated

    def step(self, action):
        # XY position
        xy_position_before = self.get_body_com("blobby")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("blobby")[:2].copy()

        # (IBN) Extra step calculation
        HP_loss = 1
        if (self.is_body_touching_floor()):
            self.increase_penalty()
            HP_loss = self._HP_loss_upon_touching_floor
        self.HP -= HP_loss

        # (IBN) Observe food
        food_eaten_this_step = self.observe_food()

        # Forward reward calcualtion
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        forward_reward = x_velocity
        # Food reward calcualtion
        # Each food eaten = +1
        food_reward = food_eaten_this_step

        # Rewards
        rewards = 0
        # rewards += forward_reward
        # rewards += healthy_reward
        # rewards += self.get_distance_from_origin()
        rewards += (1 - self.get_closest_food_distance()) / 10000
        rewards += food_reward

        # Costs
        # [!!!] COSTS ARE POSITIVE
        costs = 0
        # costs += self.get_HP_loss() / 10
        # costs = 1 if self.get_HP() <= 0 else 0
        costs += 0.05 if self.is_body_touching_floor() else 0

        # Total reward
        reward = rewards - costs

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "food_distances": self.get_food_distances_from_body(),
            "closest_food_distance": self.get_closest_food_distance(),
            "HP": self.get_HP(),
            "food_eaten_total": len(self.food_eaten_list),
            "penalty": self.get_penalty(),
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        # Blobby self-observation
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        sphereZ = [self.data.geom("sphere").xpos.copy()[2]]
        # Food observation
        closest_food = [self.get_closest_food_distance()]
        food_eaten_count = [len(self.food_eaten_list)]
        # HP observation
        currentHP = [self.get_HP()]

        return np.concatenate((position, velocity, sphereZ, closest_food, food_eaten_count, currentHP))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        # (IBN) This really should not be here, but I don't know how to overload reset()
        self.reset_custom_parameters()
        self.randomize_food_position()

        observation = self._get_obs()

        return observation
    
    # --------------------------------------------------------------
    # (IBN) Extra modification
    # --------------------------------------------------------------
    def reset_custom_parameters(self):
        self.initialize_food()
        self.initialize_HP()
        self.initialize_penalty()
    
    def initialize_food(self):
        self.food_list = []
        self.food_eaten_list = []
        self.food_progress = {}
        for i in range(self.model.nsensor):
            if (self.model.sensor(i).name.startswith("vFood")):
                self.food_list.append(self.model.sensor(i).name)
                self.food_progress[self.model.sensor(i).name] = 0

    def get_food_list_string(self):
        result = ""
        for i in self.food_list:
            result += i + " "
        return result
    
    def get_food_distances_from_body(self):
        food_distance = []
        for i in self.food_list:
            # If food is eaten, set distance to 100
            if (i in self.food_eaten_list):
                food_distance.append(100)
            else:
                food_distance.append(np.linalg.norm(self.data.geom("sphere").xpos - self.data.site(i).xpos))
        return np.array(food_distance)
    
    def observe_food(self):
        food_progress_threshold = self._food_progress_threshold

        food_eaten_this_step = 0
        for i in self.food_list:
            if (i not in self.food_eaten_list):
                if (self.data.sensor(i).data[0] > 0):
                    self.food_progress[i] += 1
                    if (self.food_progress[i] >= food_progress_threshold):
                        self.food_eaten_list.append(i)
                        self.increase_HP()
                        food_eaten_this_step += 1
        
        return food_eaten_this_step
        
    # (IBN) Food radar, because I am running out of ideas
    def get_closest_food_distance(self):
        closest_food = np.amin(self.get_food_distances_from_body())
        if (closest_food == 100):
            return 0
        return closest_food
    
    def randomize_food_position(self):
        for i in self.food_list:
            # (IBN) Hard coding: vFood1 is the food that is always at the center
            if (i == "vFood1"):
                continue
            while True:
                new_pos = self.np_random.uniform(low=-3, high=3, size=3)
                new_pos[2] = 0
                if (np.greater(np.absolute(new_pos), 1).any()):
                    break
            self.model.site(i).pos = new_pos
        
    def is_body_touching_floor(self):
        # This should have been a different function, but I am losing my mind
        # REMEMBER: MUJOCO USE Z AXIS FOR UP AND DOWN
        floor_threshold = self._floor_z_threshold
        if (self.data.geom("sphere").xpos[2] < floor_threshold):
            return True
            
        return None

    def initialize_HP(self):
        self.HP = self._initial_HP

    def increase_HP(self):
        self.HP += self._food_HP_increase

    def get_HP(self):
        return self.HP

    def initialize_penalty(self):
        self.penalty = 0

    def increase_penalty(self):
        self.penalty += 1

    def get_penalty(self):
        return self.penalty

    def get_sensor_data(self):
        return self.data.sensordata.copy()
