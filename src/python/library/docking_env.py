import numpy as np
from library.env_base import EnvStonefishRLParallel, launch_stonefish_simulator
import gymnasium as gym
import time
import gc
import os
import psutil
import yaml
from library.common import resolve_path
from library.pose_frames import invert_pose, observation_frame
from library.domain_randomization import configure_domain_randomization
from library.thruster_health import health_settings

class dsEnv(EnvStonefishRLParallel):
    """Docking task environment.

    This class is the main example for building a custom Stonefish RL task.
    For a new environment, copy this pattern and edit:

    - `build_reset_command`: which robots/objects are reset and where.
    - `reset`: task-specific counters and initial observation.
    - `step`: observation shaping, reward calculation, termination, truncation.
    - `calculate_additional_reward`: task reward terms.
    - `_check_goal_achieved` and `_get_additional_info`: success and logging.

    Observation names come from the YAML file in `env.state_config`; actuator
    command names come from `env.action_config`.
    """
    def __init__(self, rank, config,**kwargs):
        
        
        self.state_path = config["env"]["state_config"]
        self.act_path = config["env"]["action_config"]
        # Reject invalid fault settings before launching a simulator.
        with open(resolve_path(self.act_path)) as stream:
            actuator_count = len(yaml.safe_load(stream)['action_config']['specs'])
        self.use_thruster_health, self.thruster_health_settings = health_settings(config['action'], actuator_count)
        self.observation_frame = observation_frame(config["env"])
        if self.observation_frame == "docking_station":
            # Check pose fields before launching a simulator.
            with open(resolve_path(self.state_path)) as stream:
                specs = yaml.safe_load(stream)["state_config"]["specs"]
            names = [spec["output_name"] for spec in specs]
            if names[:4] != ["error_x", "error_y", "error_z", "error_yaw"] or not {"relative_roll", "relative_pitch"}.issubset(names):
                raise ValueError("Station-frame observations require the full-pose PID state config")

        
        # 1. Store timing parameters first
        self.episode_duration = config["env"]["episode_duration"]  # seconds
        self.rl_observation_freq = config["env"]["rl_observation_freq"]  # Hz
        
        
        # 3. Launch the simulator (Specific to this instance)
        self._domain_randomizer = configure_domain_randomization(config)
        self.process = None if self._domain_randomizer else launch_stonefish_simulator(rank, config)
        # Give the simulator a moment to bind the socket
        time.sleep(1.0) 
        

        """ Their should be a better way to define the action size """
        if  config["action"]["force_6Dof"]: 
        
            self.tcm = np.asarray(config["action"]["tcm"], dtype=np.float32)
            self.tcm_inv = np.linalg.pinv(self.tcm)
            action_size = None
        else: 
            self.tcm= None
            action_size = None

        self.observe_actions = config["env"]["observe_actions"]
        self.yaw_observation = str(config["env"].get("yaw_observation", "angle")).lower()
        if self.yaw_observation not in ("angle", "cos_sin"):
            raise ValueError(
                "env.yaw_observation must be either 'angle' or 'cos_sin'."
            )
        
        self.history_length = int(config["env"].get("history_length", 0))
        if self.history_length <= 0: 
            self.history_length = 0
        # Time in seconds between selected history samples. 0 keeps consecutive observations.
        self.history_gap = max(0.0, float(config["env"].get("history_gap", 0.0)))



        # 4. 
        super().__init__(action_size, env_id=rank, base_port=config["env"]["base_port"])
        self._configure_policy_observation_space()
        
        self.observation = np.array([], dtype=np.float32)
        self.observation_history = []
        self.info = {}

        # 5. Application specific init
        self.step_count = 0
        self.n_calls=0
        self.check_freq = 2100
        self.goal_achieved = False
        self.collision_recorded = 0 
        self.start_distance_factor = 1.0
        self.collision_thres = 1.0 
        self.previous_distance_error = np.zeros(3)
        self.max_dist = 10.0
        self.yaw_max = np.pi
        self.previous_yaw = np.pi
        self.velocities_history = []
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.z_offset = -0.91
        # self.yaw_offset = np.pi/2 # When the acting frame is in AUV frame
        self.yaw_offset = -3*np.pi/2 # When the acting frame is in DS frame
        self.total_yaw_reward= 0.0 

        # Goal condition 
        self.x_tol = 0.07
        self.y_tol = 0.07
        self.z_tol_start = 1.0
        self.z_tol = 0.15
        self.yaw_tol = np.deg2rad(10.0)
        self.tol_discard_rate = 0.00
        self.tol_discard_per = 0.0
        self.total_xyz = 0.0
        self.run_reward = 0.0
        self.count_achieved = 0 
        self.docking_time = 5 # Sec
        
        sim_config = config["sim"]
        self.enable_currents = sim_config["current"]
        self.current_x = sim_config["current_value"][0]
        self.current_y = sim_config["current_value"][1]
        self.currnet_uniform = sim_config["current_uniform"]
        
        legacy_noise = bool(sim_config.get("enable_noise", False))
        self.enable_offset_noise = bool(
            sim_config.get("enable_offset_noise", legacy_noise)
        )
        self.enable_random_noise = bool(
            sim_config.get("enable_random_noise", legacy_noise)
        )
        self.reward_weight=config["reward"]["weights"]
        self.gui_robot_position = None
        self.gui_robot_rotation = None
        self.gui_target_position = None
        self.gui_target_rotation = None
        self.gui_randomize_reset = True
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.current_action = np.zeros(self.action_size, dtype=np.float32)
        self.observation_noise = {
            "normal_position": np.zeros(3, dtype=np.float32),
            "unobserved_position": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
        }
    
    def update_runtime_params(self, params):
        """Apply optional GUI/test overrides without changing YAML behavior."""
        super().update_runtime_params(params)

        if "reward_weights" in params:
            self.reward_weight = list(params["reward_weights"])
        if "current_enabled" in params:
            self.enable_currents = bool(params["current_enabled"])
        if "current_value" in params:
            value = params["current_value"]
            self.current_x = float(value[0])
            self.current_y = float(value[1])
        if "current_uniform" in params:
            self.currnet_uniform = bool(params["current_uniform"])
        if "enable_offset_noise" in params:
            self.enable_offset_noise = bool(params["enable_offset_noise"])
        if "enable_random_noise" in params:
            self.enable_random_noise = bool(params["enable_random_noise"])
        if "start_distance_factor" in params:
            self.start_distance_factor = float(params["start_distance_factor"])
        if "randomize_reset" in params:
            self.gui_randomize_reset = bool(params["randomize_reset"])
        if "robot_start_position" in params:
            self.gui_robot_position = list(params["robot_start_position"])
        if "robot_start_rotation" in params:
            self.gui_robot_rotation = list(params["robot_start_rotation"])
        if "target_start_position" in params:
            self.gui_target_position = list(params["target_start_position"])
        if "target_start_rotation" in params:
            self.gui_target_rotation = list(params["target_start_rotation"])

        return self.runtime_params


    def _on_step(self) -> bool:
        self.n_calls+=1
        if self.n_calls % self.check_freq == 0:
            # print("*************** Calling GC *************")
            gc.collect()  # Force the scan
            # self.print_python_memory(f"env={self.env_id} step={self.step_count}")
            self.n_calls=0
        return True    


    def build_reset_command(self):
        """Build RESET command - specific to this application"""
        # ds_pos = [-3, -1, 5.0]
        # ds_rot = [0.0, 0.0, 0.0]
        # method one of randomization: start_distance_factor increases gradually
        if self.goal_achieved:
            if self.start_distance_factor < 1.0:
                self.start_distance_factor += 0.05
            # else:          
            #     self.start_distance_factor = np.random.random()
            
            if self.tol_discard_per > 0 : 
                # self.x_tol -= self.x_tol*self.tol_discard_rate
                # self.y_tol -= self.y_tol*self.tol_discard_rate
                self.z_tol -= self.z_tol_start*self.tol_discard_rate
                # self.yaw_tol -= self.yaw_tol*self.tol_discard_rate
                self.tol_discard_per -= self.tol_discard_rate
            


        # method two, random factor each time 
        # self.start_distance_factor = 0.1
        girona_pos = [
            0.0+ self.start_distance_factor*self.np_random.uniform(-6.0, 6.0),
            0.0+ self.start_distance_factor*self.np_random.uniform(-3.0, 3.0),
            3.5- (self.start_distance_factor*2.8 )
        ]
        girona_rot = [
            0.0+ self.start_distance_factor*self.np_random.uniform(-np.pi, np.pi)+np.pi/2,
            0.0,
            0.0
        ]

        ds_pos = [
            0.0+ self.start_distance_factor*self.np_random.uniform(-1.0, 1.0),
            0.0+ self.start_distance_factor*self.np_random.uniform(-1.0, 1.0),
            5.2
        ]
        ds_rot = [
            0.0+self.start_distance_factor*self.np_random.uniform(-np.pi, np.pi),
            0.0,
            0.0
        ]

        if not self.gui_randomize_reset:
            if self.gui_robot_position is not None:
                girona_pos = self.gui_robot_position
            if self.gui_robot_rotation is not None:
                girona_rot = self.gui_robot_rotation
            if self.gui_target_position is not None:
                ds_pos = self.gui_target_position
            if self.gui_target_rotation is not None:
                ds_rot = self.gui_target_rotation

        if self.enable_currents: 
            if self.currnet_uniform:
                current_vec = [
                    self.np_random.uniform(-self.current_x, self.current_x),
                    self.np_random.uniform(-self.current_y, self.current_y),
                    0.0,
                ]
            else: 
                current_vec = [self.current_x,self.current_y, 0.0]
        else: 
            current_vec = [0.0,0.0,0.0]

        return [
            {"name": "girona1000", "position": girona_pos, "rotation": girona_rot, "current":current_vec},
            {"name": "ds", "position": ds_pos, "rotation": ds_rot},
        ]

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        self.step_count = 0
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.observation_history = []
        
        # Distance Reset variables
        self.max_dist = 10.0
        self.previous_distance_error = np.zeros(3)
        
        # Yaw Reset variables
        self.yaw_max = np.pi
        self.collision_recorded = 0 
        self.previous_yaw = np.pi
        self.goal_achieved = False
        self.total_xyz = 0.0 
        self.run_reward = 0.0
        self.count_achieved = 0 
        self.velocities_history = []
        self.info = self._domain_randomization_info()
        self._reset_observation_noise()


        obs = self.get_sim_obser()

        return obs, self.info

    
    def step(self, action):
        """Execute step with cleaned logic"""
        self._on_step()
        self.step_count += 1
        self.current_action = np.array(action, dtype=np.float32).flatten()
        
        # Getting the robot STATE from simulator. 
        obs, reward, done, info_super = super().step(action)

        # Application logic
        
        # Calculate the main rewards for the robot current state
        additional_reward = self.calculate_additional_reward_v1()
        total_reward = reward + additional_reward
        # Update termination/truncation
        truncated = self._is_truncated()
        episode_success = False

        if self.goal_achieved:
            # Counter to stay in docking station for set of seconds, then terminate the episode.
            if self.count_achieved > self.docking_time*self.rl_observation_freq: 
                done = True
                episode_success = True
                total_reward += self.episode_duration
                print(f"[Port {self.port}] Goal achieved!, reward {additional_reward}, total reward: {self.run_reward}") 
            else: 
                self.count_achieved+=1
        else: 
            self.count_achieved = 0 

        self.run_reward += total_reward

        # Used for the smoothing reward
        self.last_action_applied = self.current_action

        self.info.update(self._get_additional_info())
        # Success requires completing the docking hold, not briefly reaching the goal.
        self.info['is_success'] = episode_success
        obs = self._build_policy_observation(obs)
        if self.observe_actions:
            obs = np.concatenate((obs,self.current_action))
        
               
        self.concatenate_history(obs,history_gap=self.history_gap)

        observation = np.round(self.observation, 3)
        if self.use_thruster_health:
            observation[-len(self.thruster_health):] = self.thruster_health

        # print(f"[observation {len(observation)}]: {observation}")
        return observation, total_reward, done, truncated, self.info

    
    # uses the default setup v1
    def calculate_additional_reward_v1(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        error = self._get_state_by_pattern("error").copy()
        velocity = self._get_state_by_pattern("velocity").copy()
        x_err = error[0] + self.x_offset
        y_err = error[1] + self.y_offset
        z_err = error[2] + self.z_offset

        xy_error = np.linalg.norm([x_err, y_err])
        distance_error = np.linalg.norm([x_err, y_err, z_err])
        z_abs_error = np.clip(z_err,0.01,5)/5

        yaw_err = wrap_to_pi(error[3] + self.yaw_offset)

        visible = self._auv_observed_ds()
        # Visibility penalty 
        # reward_visible = 0.0 if visible else -3 

        # xy penalty 
        xy_error = np.linalg.norm([x_err, y_err])
        # reward_xy = np.clip(1/(xy_error+ 1e-6),0,1.0) * self.reward_weight[0]
        reward_xy = -np.clip(xy_error/5,0.05,5.0) * self.reward_weight[0]
        # z_penalty
        reward_z = 0.0 
        reward_z = -np.clip (z_abs_error,0,1.0)* self.reward_weight[1]

        # Yaw reward: 
        reward_yaw = 0.0
        reward_yaw = -abs(yaw_err) / np.pi * self.reward_weight[2]

        # docking reward
        docking_accuracy =  (0.5*(xy_error) ** 2) / (z_abs_error )
        docking_accuracy_reward = -np.clip(docking_accuracy,0,5) * self.reward_weight[3]
        # --- checking formula: 
        # docking_accuracy_reward = - np.clip(self.current_action[2]*xy_error/2,0,5)
                
        # Smoothing reward 
        # smoothing_reward = 0 
        action_difference = abs(self.current_action - self.last_action_applied)
        # action_value = np.linalg.norm(self.current_action[:3])+np.linalg.norm(self.last_action_applied[4:6]) 
        smoothing_reward = -np.clip(np.linalg.norm(action_difference),0,0.5)* self.reward_weight[4]
        # relative velocity
        velocity_reward = 0.0 
        mean_velocity = np.linalg.norm(velocity[:2])

        velocity_reward = -np.clip(mean_velocity/distance_error,0,1.0) * self.reward_weight[5]

        # Collision reward
        collision_reward = 0.0
        collided, acceleration = self.detect_collision()
        collision_reward = -np.clip (acceleration**2,0,20) * self.reward_weight[6]
        # if collided: 
        #     self.collision_recorded +=1
        #     collision_reward -= 2*self.rl_observation_freq 

        reward =  reward_xy + reward_z + reward_yaw + docking_accuracy_reward + smoothing_reward +velocity_reward + collision_reward # this value makes the weight moving the ball equivalent to reaching the ball
        
        if self._check_goal_achieved() and visible:
            reward = self.rl_observation_freq
            self.goal_achieved = True
        else:
            self.goal_achieved = False
        
        self.info.update({
            "reward_xy": reward_xy,
            "reward_z": reward_z,
            "reward_yaw": reward_yaw, 
            "docking_accuracy_reward": docking_accuracy_reward,
            "smoothing_reward": smoothing_reward, 
            "mean_velocity": mean_velocity,
            "velocity_reward": velocity_reward,
            "collision_reward": collision_reward, 
            "acceleration": acceleration, 
            })
        

        reward /= self.rl_observation_freq # making reward homogenous regardless of the RL frequency   

        return reward


    # uses the default setup v3
    def calculate_additional_reward_v3(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        error = self._get_state_by_pattern("error").copy()
        velocity = self._get_state_by_pattern("velocity").copy()
        x_err = error[0] + self.x_offset
        y_err = error[1] + self.y_offset
        z_err = error[2] + self.z_offset

        xy_error = np.linalg.norm([x_err, y_err])
        distance_error = np.linalg.norm([x_err, y_err, z_err])
        z_abs_error = np.clip(z_err,0.01,5)

        yaw_err = wrap_to_pi(error[3] + self.yaw_offset)

        visible = self._auv_observed_ds()
        # Visibility penalty 
        # reward_visible = 0.0 if visible else -3 

        # xy penalty 
        xy_error = np.linalg.norm([x_err, y_err])
        reward_xy = -np.clip(xy_error,0.05,5.0) * self.reward_weight[0]

        # z_penalty
        # reward_z = 0.0
        reward_z = -abs(z_abs_error) * self.reward_weight[1]

        # Yaw reward: 
        reward_yaw = 0.0
        reward_yaw = -abs(yaw_err / np.pi *(1/reward_xy) )* self.reward_weight[2]

        # docking accuracy
        if xy_error > z_abs_error-1: 
            docking_accuracy_reward = -abs(np.clip((0.5*(xy_error) ** 2) / (z_abs_error),0,100)) * self.reward_weight[3]
        else: 
            docking_accuracy_reward = 0.0                

        reward =  reward_xy + reward_z + reward_yaw + docking_accuracy_reward 

        if self._check_goal_achieved() and visible:
            reward = self.rl_observation_freq
            self.goal_achieved = True
        else:
            self.goal_achieved = False
        
        self.info.update({
            "reward_xy": reward_xy,
            "reward_z": reward_z,
            "z_error": z_err,
            "reward_yaw": reward_yaw, 
            "docking_accuracy_reward": docking_accuracy_reward,
          
            })
        

        reward /= self.rl_observation_freq # making reward homogenous regardless of the RL frequency   

        return reward

    # uses the default setup v1-2
    def calculate_additional_reward_v1_2(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        error = self._get_state_by_pattern("error").copy()
        velocity = self._get_state_by_pattern("velocity").copy()
        x_err = error[0] + self.x_offset
        y_err = error[1] + self.y_offset
        z_err = error[2] + self.z_offset

        xy_error = np.linalg.norm([x_err, y_err])
        distance_error = np.linalg.norm([x_err, y_err, z_err])
        z_abs_error = np.clip(z_err,0.01,5)/5

        yaw_err = wrap_to_pi(error[3] + self.yaw_offset)

        visible = self._auv_observed_ds()
        # Visibility penalty 
        # reward_visible = 0.0 if visible else -3 

        # xy penalty 
        xy_error = np.linalg.norm([x_err, y_err])
        reward_xy = -np.clip(xy_error/5,0,1.0) * self.reward_weight[0]

        # z_penalty
        reward_z = 0.0 
        reward_z = -np.clip (z_abs_error,0,1.0)* self.reward_weight[1]

        # Yaw reward: 
        reward_yaw = 0.0
        reward_yaw = -abs(yaw_err) / np.pi * self.reward_weight[2]
                
        # docking reward
        docking_accuracy =  (0.5*(xy_error) ** 2) / (z_abs_error )
        docking_accuracy_reward = -np.clip(docking_accuracy,0,5) * self.reward_weight[3]
        # --- checking formula: 
        # docking_accuracy_reward = - np.clip(self.current_action[2]*xy_error/2,0,5)
        
        # Smoothing reward 
        # smoothing_reward = 0 
        action_difference = abs(self.current_action - self.last_action_applied)
        # action_value = np.linalg.norm(self.current_action[:3])+np.linalg.norm(self.last_action_applied[4:6]) 
        smoothing_reward = -np.clip(np.linalg.norm(action_difference),0,0.5)* self.reward_weight[4]
        # relative velocity
        velocity_reward = 0.0

        d_approach = 2.5 # meter
        v_max_dock = 0.15 # m/s
        linear_speed = np.linalg.norm(velocity[:3])
        angular_speed = np.linalg.norm(velocity[3:6])
        if distance_error < d_approach: 
            distance_factor = (d_approach-distance_error)/d_approach
            overspeed = max(linear_speed + angular_speed - v_max_dock,0.0)
            velocity_reward = np.clip(-overspeed * distance_factor,-1.0,0) * self.reward_weight[5]



        # Collision reward
        collision_reward = 0.0
        collided, acceleration = self.detect_collision()
        collision_reward = -np.clip (acceleration**2,0,60) * self.reward_weight[6]
        # if collided: 
        #     self.collision_recorded +=1
        #     collision_reward -= 2*self.rl_observation_freq 

        reward =  reward_xy + reward_z + reward_yaw + docking_accuracy_reward + smoothing_reward + velocity_reward + collision_reward # this value makes the weight moving the ball equivalent to reaching the ball
        
        if self._check_goal_achieved() and visible:
            reward += self.rl_observation_freq * 2
            self.goal_achieved = True
        else:
            self.goal_achieved = False
        
        self.info.update({
            "reward_xy": reward_xy,
            "reward_z": reward_z,
            "z_error": z_err,
            "reward_yaw": reward_yaw, 
            "docking_accuracy_reward": docking_accuracy_reward,
            "smoothing_reward": smoothing_reward, 
            "velocity_reward": velocity_reward,
            "collision_reward": collision_reward, 
            "acceleration": acceleration, 
            "linear_speed": linear_speed,
            "angular_speed": angular_speed
            })
        

        reward /= self.rl_observation_freq # making reward homogenous regardless of the RL frequency   

        return reward
    
    # Uses the simplified setup
    def calculate_additional_reward_v2(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        error = self._get_state_by_pattern("error").copy()
        velocity = self._get_state_by_pattern("velocity").copy()
        x_err = error[0] + self.x_offset
        y_err = error[1] + self.y_offset
        z_err = error[2] + self.z_offset

        xyz_error = np.linalg.norm([x_err, y_err,z_err])
        visible = self._auv_observed_ds()

        # xyz penalty 
        reward_xyz = -np.clip(xyz_error/5,0,1.0) * self.reward_weight[0]

        self.total_xyz += abs(reward_xyz)

        # docking reward
        xy_error = np.linalg.norm([x_err, y_err])
        docking_accuracy =  (0.25*(xy_error) ** 2) / (abs(z_err) + 0.01)
        docking_accuracy_reward = -np.clip(docking_accuracy,0,5) * self.reward_weight[1]
                
        # Smoothing reward 
        # smoothing_reward = 0 
        action_difference = abs(self.current_action[:3] - self.last_action_applied[:3])
        # action_value = np.linalg.norm(self.current_action[:3])+np.linalg.norm(self.last_action_applied[4:6]) 
        smoothing_reward = -np.clip(np.linalg.norm(action_difference)/3,0,0.5)* self.reward_weight[2]
    
        # Collision reward
        collision_reward = 0.0
        collided, acceleration = self.detect_collision()
        collision_reward = -np.clip (acceleration**2,0,20) * self.reward_weight[3]
        
        # Velocity Control
        velocity_reward = 0 
        mean_velocity = np.linalg.norm(velocity)
        reward_velocity = -np.clip(mean_velocity/(xyz_error+0.5),0,1.0) * self.reward_weight[4]
        # print(f"velocity: {velocity}, mean velocity: {mean_velocity}, reward_velocity: {velocity_reward}")
        # if collided: 
        #     self.collision_recorded +=1
        #     collision_reward -= 2*self.rl_observation_freq 

        reward =  reward_xyz + docking_accuracy_reward + smoothing_reward + collision_reward + reward_velocity # this value makes the weight moving the ball equivalent to reaching the ball
         
        if self._check_goal_achieved() and visible:
            reward = 100
            self.goal_achieved = True
        else:
            self.goal_achieved = False
        
        self.info.update({
            "reward_xyz": reward_xyz,
            "docking_accuracy_reward": docking_accuracy_reward,
            "smoothing_reward": smoothing_reward, 
            "collision_reward": collision_reward, 
            "reward_velocity": reward_velocity,
            "acceleration": acceleration, 
            })
        

        # reward /= self.rl_observation_freq # making reward homogenous regardless of the RL frequency   

        return reward
    
    

    def _check_goal_achieved(self):
        error = self._get_state_by_pattern("error").copy()

        x_err = error[0] + self.x_offset
        y_err = error[1] + self.y_offset
        z_err = error[2] + self.z_offset
        yaw_err = wrap_to_pi(error[3] + self.yaw_offset)


        pos_success = (
            abs(x_err) < self.x_tol and
            abs(y_err) < self.y_tol and
            abs(z_err) < self.z_tol
        )

        yaw_success = abs(yaw_err) < self.yaw_tol

        return pos_success and yaw_success
    
    def concatenate_history(self, obs=None, history_gap=0): 
        """Concatenate observation history: 

            observation_history stores only the needed items for the observation,
            this can be improved by storing more history steps. Then select the needed
            items for the observation from the history.
        """
        # can be done once in SELF    
        if history_gap >0 : 
            extra_steps = max(1, int(round(history_gap * self.rl_observation_freq)))
            desired_positions = []

            if not isinstance(obs, np.ndarray):
                observation_history = np.zeros(((self.history_length*extra_steps)+1,self.obs_size))
                self.observation_history = list(observation_history)
            
            if isinstance(obs, np.ndarray):
                self.observation_history.append(obs)

            for i in range(self.history_length+1):
                desired_positions.append(-1-(i*extra_steps))

            if len(self.observation_history) > (self.history_length*extra_steps)+1:
                self.observation_history.pop(0)

            self.observation= np.array(self.observation_history)[desired_positions].flatten()
            # print(f"self.observation: {len(self.observation)}, desired_positions: {desired_positions}, history_length: {self.history_length}, extra_steps: {extra_steps}")

        else : 
            # For reset and initialize
            if not isinstance(obs, np.ndarray):
                observation_history = np.zeros((self.history_length,self.obs_size))
                self.observation_history = list(observation_history)

            if isinstance(obs, np.ndarray):
                self.observation_history.append(obs)
            extra_steps = 0

            if len(self.observation_history) > (self.history_length+1):
                self.observation_history.pop(0)
        
            try:
                self.observation = np.concatenate((self.observation_history), axis=0)
            except ValueError as e:
                # print(f"Error concatenating history: {e}")
                self.observation = np.zeros(self.obs_size * (1 + self.history_length), dtype=np.float32)
            # print(f"observation size after concatenation: {len(self.observation)}/{self.total_obs_size}")

        history_size = self.obs_size * (1 + self.history_length)
        length = history_size - len(self.observation)
        if length > 0:
            self.observation = np.concatenate((self.observation, np.zeros(length)))

        # Episode-level health is a suffix, never stored in observation_history.
        if self.use_thruster_health:
            self.observation = np.concatenate((self.observation, self.thruster_health))
        self.observation = self.observation.astype(np.float32)
        return self.observation

    def _configure_policy_observation_space(self):
        """Expose yaw error as cos/sin without changing the raw simulator state."""
        self.policy_state_names = []
        for name in self.state_names:
            if name == "error_yaw" and self.yaw_observation == "cos_sin":
                self.policy_state_names.extend(["error_yaw_cos", "error_yaw_sin"])
            else:
                self.policy_state_names.append(name)

        if self.observe_actions:
            self.observation_names = self.policy_state_names + [
                f"action_{self.action_names[index]}"
                if index < len(self.action_names)
                else f"action_{index}"
                for index in range(self.action_size)
            ]
        else:
            self.observation_names = list(self.policy_state_names)

        self.obs_size = len(self.observation_names)
        self.total_obs_size = self.obs_size * (1 + self.history_length)
        if self.use_thruster_health:
            self.total_obs_size += len(self.thruster_health)
        # print(f"total_obs_size: {self.total_obs_size}, obs_size: {self.obs_size}, history_length: {self.history_length}")
        self.total_observation_names = []
        for history_index in range(1 + self.history_length):
            suffix = "" if history_index == 0 else f"_history_{history_index}"
            self.total_observation_names.extend(
                f"{name}{suffix}" for name in self.observation_names
            )
        if self.use_thruster_health:
            self.total_observation_names.extend(f"thruster_health_{name}" for name in self.action_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_size,), dtype=np.float32
        )

    def _build_policy_observation(self, raw_obs):
        """Convert raw simulator state into the observation returned to the agent."""
        obs = np.asarray(raw_obs, dtype=np.float32).copy()
        if len(obs) < 4:
            return obs

        if self.observation_frame == "docking_station":
            roll_index = self.state_names.index("relative_roll")
            pitch_index = self.state_names.index("relative_pitch")
            pose, _ = invert_pose([*obs[:3], obs[roll_index], obs[pitch_index], obs[3]])
            # Invert the RAW pose first. Noise below is now in station coordinates.
            obs[:3], obs[3] = pose[:3], pose[5]
            obs[roll_index], obs[pitch_index] = pose[3:5]
        else:
            obs[2] += self.z_offset
            obs[3] = wrap_to_pi(obs[3] + self.yaw_offset)

        if self.enable_offset_noise:
            obs = self._apply_observation_offset_noise(obs)
        
        if self.enable_random_noise:
            obs = self._apply_observation_random_noise(obs)

        obs[:3] = np.clip(obs[:3], -5.0, 5.0) / 5.0
        yaw = wrap_to_pi(obs[3])
        if self.yaw_observation == "angle":
            obs[3] = yaw / np.pi
            return obs.astype(np.float32)

        encoded_obs = np.concatenate(
            (
                obs[:3],
                np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32),
                obs[4:],
            )
        )
        return encoded_obs.astype(np.float32)

    def _reset_observation_noise(self):
        """Sample fixed observation noise for one episode.

        The same noise direction is reused during the episode, then scaled by
        the current docking error so it fades as the AUV gets close to the DS.
        """
        self.observation_noise = {
            "accumilated_noise": self.np_random.uniform(-0.01, 0.01, 3).astype(np.float32),
            "unobserved_position": self.np_random.uniform(-0.01, 0.01, 3).astype(np.float32),
            "yaw": float(self.np_random.uniform(-0.1, 0.1)),
        }

    def _normalized_position_error(self, divider=5.0):
        """Return a 0..1 scale from the current 3D docking position error."""
        position_error = np.linalg.norm(self._get_state_by_pattern("error")[:3])/ divider
        return np.clip(position_error , 0.0, 1.0)

    def _apply_observation_offset_noise(self, obs):
        """Apply the per-episode observation noise with distance-based fading."""
        error_scale = self._normalized_position_error()
        normal_scale = error_scale / 2.0

        if not self._auv_observed_ds():
            self.observation_noise["accumilated_noise"] += self.observation_noise["unobserved_position"] * (1/self.rl_observation_freq)
        else: 
            self.observation_noise["accumilated_noise"] *= 0.1 * (1/self.rl_observation_freq) # decay the noise when observed
        
        obs[:3] += self.observation_noise["accumilated_noise"]

        obs[3] = np.clip(
            obs[3] + self.observation_noise["yaw"] * normal_scale,
            -np.pi,
            np.pi,
        )

        return obs

    def _apply_observation_random_noise(self, obs):
        """Apply per-step camera/pose-estimation fluctuation when DS is observed."""
        
        if self._auv_observed_ds():
            error_scale = self._normalized_position_error()
            random_position_noise = self.np_random.uniform(-0.1, 0.1, 3).astype(np.float32) * error_scale
            random_yaw_noise = float(self.np_random.uniform(-0.05, 0.05)) * error_scale
            obs[:3] += random_position_noise
            obs[3] = wrap_to_pi(obs[3] + random_yaw_noise)
        
        return obs

    def get_sim_obser(self):
        """Build observation: state vector + last action + zeros from history"""
        return self.concatenate_history(history_gap=self.history_gap)

    def detect_collision(self, acceleration_threshold=3):
        """
        Detect collision based on sudden change in velocity.

        A collision is detected when the acceleration magnitude exceeds
        `acceleration_threshold`.

        Parameters
        ----------
        acceleration_threshold : float
            Acceleration threshold in m/s^2.

        Returns
        -------
        boolq
            True if collision is detected, otherwise False.
        """

        current_velocity = self._get_state_by_pattern("velocity")

        # Append current velocity
        self.velocities_history.append(current_velocity)

        # Keep only the last two velocity vectors
        self.velocities_history = self.velocities_history[-2:]

        # Need two velocity vectors to compute acceleration
        if self.step_count <= 2:
            return False,0

        previous_velocity = self.velocities_history[0]
        current_velocity = self.velocities_history[1]

        # Change in velocity
        delta_velocity = current_velocity - previous_velocity
        acceleration = delta_velocity * self.rl_observation_freq

        acceleration_magnitude = np.linalg.norm(acceleration)
        
        return acceleration_magnitude > acceleration_threshold, acceleration_magnitude


    def _get_state_by_pattern(self, pattern, default=0.0):
        """Get observation value by name pattern"""
        # print("[DEBUG] self.state_names", self.state_names)
        value = []
        for i, name in enumerate(self.state_names):

            if pattern in name and i < len(self.state):
                # print("Matched")
                value.append(self.state[i])
        return np.array(value) if len(value)>0 else default

    def _distance_to_target(self, robot_pos):
        """Distance to target (ds)"""
        # target_pos = np.array([0.0, 0.0, 5.0])
        target_pos = self._get_state_by_pattern("ds_pose")[:3]
        return np.linalg.norm(robot_pos - target_pos)

    def _is_terminated(self):
        """Application-specific termination conditions"""
        robot_pos = self._get_state_by_pattern("auv_pose")
        return False

    def _is_truncated(self):
        """Cleaned truncation logic""" 
        return self.step_count/self.rl_observation_freq >= self.episode_duration
    
    def _auv_observed_ds(self, a=3.0, b=4.0, h=5.0, eps=1e-9):
        """Check if AUV has observed the docking station"""
        error = self._get_state_by_pattern("error").copy()        
        # check if AUV lies in DS cone field of view
          
        x = error[0]
        y = error[1]
        z = error[2]
        yaw = wrap_to_pi(error[3] + self.yaw_offset)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        x_local = cos_yaw * x + sin_yaw * y
        y_local = -sin_yaw * x + cos_yaw * y
        z_local = z

        # Check depth range
        if z_local < -eps or z_local > h + eps:
            return False

        # At the apex, only the origin belongs to the cone
        if abs(z_local) < eps:
            return abs(x_local) < eps and abs(y_local) < eps

        # Ellipse size grows linearly with distance z
        a_z = a * z_local / h
        b_z = b * z_local / h

        value = (x_local / a_z) ** 2 + (y_local / b_z) ** 2

        return value <= 1.0 + eps
    


    def _get_additional_info(self):
        """Additional info for this application"""
        # robot_pos = self._get_state_by_pattern("auv_pose")
        
        
        return {
            # "distance_to_target": self._distance_to_target(robot_pos[:3]),
            "error": self._get_state_by_pattern("error").tolist(),
            "ds_observed": self._auv_observed_ds(),
            "step": self.step_count
        }

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
