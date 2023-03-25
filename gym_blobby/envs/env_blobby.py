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
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=False,
        healthy_z_range=(0.2, 2.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # (IBN) Observation space
        # Joint position:
        # - Free joint = 7
        # - Fixed joint = 1
        # Joint velocity:
        # - Free joint = 6
        # - Fixed joint = 1
        # Blobby has 1 free joint and 12 fixed joint
        # obs_space += (7 + 12 * 1) + (6 + 12 * 1) = 37
        # [!!!] HARD CODING BECAUSE THE MODEL IS INITIALIZED AFTER THE OBSERVATION SPACE
        # Food count = 1 (closest food)
        # obs_space += 1 = 38
        # Sphere xpos = 3
        # obs_space += 3 = 41
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
        self.initialize_food()
        self.initialize_HP()

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = False

        # Healthy termination (legacy)
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False

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
        self.step_blobby()

        # (IBN) Observe food
        food_found = self.on_body_touches_food()
        if ((food_found is not None) and (food_found not in self.food_eaten_list)):
            self.food_eaten_list.append(food_found)
            self.increase_HP()

        # Forward reward calcualtion
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        forward_reward = x_velocity
        # Healthy reward calculation
        healthy_reward = self.healthy_reward
        # Food reward calcualtion
        # Each food eaten = +2
        food_reward = len(self.food_eaten_list)

        # Rewards
        rewards = 0
        # rewards += forward_reward
        # rewards += healthy_reward
        # rewards += self.get_distance_from_origin()
        # rewards += self.get_timeStep() / 100000
        rewards += (1 - self.get_closest_food_distance()) / 10
        rewards += food_reward

        # Costs
        costs = 0
        # costs = ctrl_cost = self.control_cost(action)
        # costs += self.get_HP_loss() / 10
        # costs = 1 if self.get_HP() <= 0 else 0
        costs += self.get_penalty() / 100

        reward = rewards - costs

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin_SB3": np.linalg.norm(xy_position_after, ord=2),
            "distance_from_origin_IBN": self.get_distance_from_origin(),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "food_distances": self.get_food_distances_from_body(),
            "closest_food_distance": self.get_closest_food_distance(),
            "HP": self.get_HP(),
            "timeStep": self.get_timeStep(),
            "food_reward": food_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # food = self.get_food_distances_from_body()
        food = [self.get_closest_food_distance()]
        sphere = self.data.geom("sphere").xpos.copy()

        # return np.concatenate((position, velocity, food))
        return np.concatenate((position, velocity, food, sphere))

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
    def initialize_food(self):
        self.food_eaten_list = []
        self.food_list = []
        for i in range(self.model.ngeom):
            if (self.model.geom(i).name.startswith("food")):
                self.food_list.append(self.model.geom(i).name)

    def get_food_list(self):
        result = ""
        for i in self.food_list:
            result += i + " "
        return result
    
    def get_food_distances_from_body(self):
        food_distance = []
        for i in self.food_list:
            # If food is eaten, set distance to -1
            if (i in self.food_eaten_list):
                food_distance.append(10)
            else:
                food_distance.append(np.linalg.norm(self.data.geom("sphere").xpos - self.data.geom(i).xpos))
        return np.array(food_distance)
    
    def get_distance_from_origin(self):
        return np.linalg.norm(self.data.geom("sphere").xpos, ord=2)
    
    def on_body_touches_food(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body = None
            if self.data.geom(contact.geom1).name.startswith("sphere"):
                body = contact.geom1
            elif self.data.geom(contact.geom2).name.startswith("sphere"):
                body = contact.geom2
            food = None
            if self.data.geom(contact.geom1).name.startswith("food"):
                food = contact.geom1
            elif self.data.geom(contact.geom2).name.startswith("food"):
                food = contact.geom2

            if (body is not None) and (food is not None):
                return self.data.geom(food).name
            
            return None
        
    def is_body_touching_floor(self):
        # for i in range(self.data.ncon):
        #     contact = self.data.contact[i]
        #     body = None
        #     if self.data.geom(contact.geom1).name.startswith("sphere"):
        #         body = contact.geom1
        #     elif self.data.geom(contact.geom2).name.startswith("sphere"):
        #         body = contact.geom2
        #     floor = None
        #     if self.data.geom(contact.geom1).name.startswith("floor"):
        #         floor = contact.geom1
        #     elif self.data.geom(contact.geom2).name.startswith("floor"):
        #         floor = contact.geom2

        #     if (body is not None) and (floor is not None):
        #         return True

        # This should have been a different function, but I am losing my mind
        # REMEMBER: MUJOCO USE Z AXIS FOR UP AND DOWN
        if (self.data.geom("sphere").xpos[2] < 0.25):
            return True
            
        return None

    
    def reset_custom_parameters(self):
        self.food_eaten_list = []
        self.initialize_HP()
        self.initialize_penalty()
        self.initialize_timeStep()

    def initialize_HP(self):
        # (IBN) THIS IS HARD CODING !!!
        self.HP = 250

    def increase_HP(self):
        # (IBN) THIS IS HARD CODING !!!
        self.HP += 100

    def get_HP(self):
        return self.HP

    def initialize_penalty(self):
        self.penalty = 0

    def increase_penalty(self):
        self.penalty += 1

    def get_penalty(self):
        return self.penalty
    
    def step_blobby(self):
        # (IBN) TODO: REFACTOR THIS
        HP_loss = 1
        if (self.is_body_touching_floor()):
            self.increase_penalty()
            HP_loss = 20

        self.HP -= HP_loss
        self.timeStep += 1

    def initialize_timeStep(self):
        # I am sure that there is a built-in variable for this,
        # but I don't know how to access it.
        self.timeStep = 0

    def get_timeStep(self):
        return self.timeStep
    
    # (IBN) Food radar, because I am running out of ideas
    def get_closest_food_distance(self):
        # food_distance = []
        # for i in self.food_list:
        #     if (i in self.food_eaten_list):
        #         continue
        #     else:
        #         food_distance.append(np.linalg.norm(self.data.geom("sphere").xpos - self.data.geom(i).xpos))
        
        # if (len(food_distance) == 0):
        #     return 0
        # return min(food_distance)

        closest_food = np.amin(self.get_food_distances_from_body())
        if (closest_food == 10):
            return 0
        return closest_food
    
    def randomize_food_position(self):
        for i in self.food_list:
            if (i == "food1"):
                continue
            new_pos = self.np_random.uniform(low=-2.5, high=2.5, size=3)
            new_pos[2] = 0
            self.model.geom(i).pos = new_pos
