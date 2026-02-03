import numpy as np
import json
from EnvStonefishRL import EnvStonefishRLParallel,launch_stonefish_simulator
import gymnasium as gym
import time

class dsEnv(EnvStonefishRLParallel):
    def __init__(self, rank, config,**kwargs):
        
        
        obs_path = config["env"]["observation_config"]
        act_path = config["env"]["action_config"]

        
        # 1. Store timing parameters first
        self.episode_duration = config["env"]["episode_duration"]  # seconds
        self.rl_observation_freq = config["env"]["rl_observation_freq"]  # Hz
        
        
        # 3. Launch the simulator (Specific to this instance)
        self.process = launch_stonefish_simulator(rank, config)            
        # Give the simulator a moment to bind the socket
        time.sleep(1.0) 
        
        # 4. Initialize parent (ZMQ connection)
        super().__init__(obs_path, act_path, rank, config["env"]["base_port"])

        # 5. Application specific init
        self.step_counter = 0
        self.goal_achieved = False
        self.start_distance_factor = 0.0
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.current_action = np.zeros(self.action_size, dtype=np.float32)
        
        

    def build_reset_command(self):
        """Build RESET command - specific to this application"""
        # ds_pos = [-3, -1, 5.0]
        # ds_rot = [0.0, 0.0, 0.0]
        # method one of randomization: start_distance_factor increases gradually
        # if self.goal_achieved:
        #     if self.start_distance_factor < 1.0:
        #         self.start_distance_factor += 0.05

        # method two, random factor each time 
        self.start_distance_factor = np.random.random()
        # self.start_distance_factor = 0.1
        girona_pos = [
            0.0+ self.start_distance_factor*self.np_random.uniform(-6.0, 6.0),
            0.0+ self.start_distance_factor*self.np_random.uniform(-3.0, 3.0),
            4.0- (self.start_distance_factor*2.8 )
        ]
        girona_rot = [
            0.0+ self.start_distance_factor*self.np_random.uniform(-np.pi, np.pi),
            0.0,
            0.0
        ]

        ds_pos = [
            0.0+ self.start_distance_factor*self.np_random.uniform(-1.0, 1.0),
            0.0+ self.start_distance_factor*self.np_random.uniform(-1.0, 1.0),
            5.5
        ]
        ds_rot = [
            0.0+0.5*self.start_distance_factor*self.np_random.uniform(-np.pi, np.pi),
            0.0,
            0.0
        ]

        current_vec = [self.np_random.uniform(-.1, 0.1),self.np_random.uniform(-0.1, 0.1), 0.0]

        return [
            {"name": "girona500", "position": girona_pos, "rotation": girona_rot, "current":current_vec},
            {"name": "ds", "position": ds_pos, "rotation": ds_rot},
        ]

    def reset(self, seed=None, options=None):
        """Reset environment"""
        command = self.build_reset_command()
        reset_command = "RESET:" + json.dumps(command) + ";"
        
        # Use parent reset but send our specific reset command
        response = self.send_command(reset_command)
        self._process_observation_vector(response)
        
        self.step_counter = 0
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.goal_achieved = False
        obs = self.get_observation()
        info = {}
        
        return obs, info

    def get_observation(self):
        """Build observation: state vector + last action"""
        obs = []
        
        # Add the observation vector from C++
        if len(self.state) > 0:
            obs.extend(self.state.tolist())
        else:
            # Fallback: zeros
            obs.extend([0.0] * self.observation_size)
        
        # Add last action
        # obs.extend(self.last_action_applied.tolist())

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute step with cleaned logic"""
        self.step_counter += 1
        self.current_action = np.array(action, dtype=np.float32).flatten()
        
        obs, reward, terminated, truncated, info = super().step(action)

        # Application logic
        if not self._auv_observed_ds():
            obs[0:3] += np.random.random(3) # mark as out of sight
            additional_reward = -6.0 # Penalty for losing sight of the docking station
        else:
            additional_reward = self.calculate_additional_reward()
            
        # Update termination/truncation
        terminated = terminated or self._is_terminated()
        truncated = truncated or self._is_truncated()
            
        
        # Goal check
        if self.goal_achieved:
            print(f"[Port {self.port}] Goal achieved!")
            terminated = True
        
        if truncated:
            additional_reward -= 10.0  # Penalty for timeout
            
        total_reward = reward + additional_reward
        self.last_action_applied = self.current_action
        info.update(self._get_additional_info())
        return np.round(obs,2), total_reward, terminated, truncated, info
    

    def calculate_additional_reward(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        robot_pos = self._get_observation_by_pattern("auv_pose")
        ds_pose = self._get_observation_by_pattern("ds_pose")
        distance_x = robot_pos[0] - ds_pose[0]
        distance_y = robot_pos[1] - ds_pose[1]
        distance_z = robot_pos[2] - ds_pose[2]
        Yaw_error = robot_pos[3] - ds_pose[3]


        reward_dist = -abs(distance_x)-abs(distance_y)-(abs(distance_z)-1.25)*0.5
        reward_yaw = np.exp(-2*abs(Yaw_error))-1
        smoothing_reward = - abs(self.current_action - self.last_action_applied).sum()/abs(2*np.ones(self.action_size)).sum()
        # print(f"[Port {self.port}], reward_yaw {reward_yaw:.2f} Action smoothness penalty: {smoothing_reward:.4f}")        

        
        # print(f"[DEBUG] dist_to_target: {dist_to_target}, dist_to_goal: {dist_to_goal}, collision_flag: {collision_flag}")
        reward = reward_dist  + reward_yaw + smoothing_reward# this value makes the weight moving the ball equivalent to reaching the ball

        if abs(distance_z) > 1.27 or abs(distance_x) > 0.15 or abs(distance_y) > 0.15 : 
            self.goal_achieved = False
        else:
            self.goal_achieved = True
            reward += 500.0  # Large reward for reaching the goal
        print(f"[Port {self.port}] Distance to target: {distance_x:.2f} {distance_y:.2f} {distance_z:.2f} m , reward: {reward:.2f}", end='\r')

        # print(f"[DEBUG] Calculated additional reward: {reward}")    
        return reward

    def _get_observation_by_pattern(self, pattern, default=0.0):
        """Get observation value by name pattern"""
        # print("[DEBUG] self.observation_names", self.observation_names)
        value = []
        for i, name in enumerate(self.observation_names):

            if pattern in name and i < len(self.state):
                # print("Matched")
                value.append(self.state[i])
        return np.array(value) if len(value)>0 else default

    def _distance_to_target(self, robot_pos):
        """Distance to target (ds)"""
        # target_pos = np.array([0.0, 0.0, 5.0])
        target_pos = self._get_observation_by_pattern("ds_pose")[:3]
        return np.linalg.norm(robot_pos - target_pos)



    def _is_terminated(self):
        """Application-specific termination conditions"""
        robot_pos = self._get_observation_by_pattern("auv_pose")
        collision_flag = self._get_observation_by_pattern("collision", default=0.0)
        # Terminate if collision
        if collision_flag > 0.5:
            return True
            
        return False

    def _is_truncated(self):
        """Cleaned truncation logic""" 
        return self.step_counter/self.rl_observation_freq >= self.episode_duration
    
    def _auv_observed_ds(self):
        """Check if AUV has observed the docking station"""
        ds_pos = self._get_observation_by_pattern("ds_pose")[:3]
        auv_pos = self._get_observation_by_pattern("auv_pose")[:3]
        
        # check if AUV lies in DS cone field of view
        k = 0.64 # considering the cone size of (2,2,4.4) 
        xy_plane_distance = np.linalg.norm(ds_pos[:2]-auv_pos[:2])
        z_distance = k*abs(ds_pos[2]-auv_pos[2])
        return xy_plane_distance <= z_distance 


    def _get_additional_info(self):
        """Additional info for this application"""
        robot_pos = self._get_observation_by_pattern("auv_pose")
        return {
            "distance_to_target": self._distance_to_target(robot_pos[:3]),
            "ds_position": self._get_observation_by_pattern("ds_pose").tolist(),
            "ds_observed": self._auv_observed_ds(),
            "step": self.step_counter
        }