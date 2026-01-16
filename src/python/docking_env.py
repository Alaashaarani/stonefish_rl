import numpy as np
import json
from EnvStonefishRL import EnvStonefishRLParallel,launch_stonefish_simulator
import gymnasium as gym
import time

class dsEnv(EnvStonefishRLParallel):
    def __init__(self, observation_config_path, action_config_path,
                 resolution=300, 
                 env_id=0,
                 base_port=5555,
                 episode_duration=120,
                 simulation_frequency=50,
                 rl_frequency=10,
                 graphical=False,
                 **kwargs):
        
        # 1. Store timing parameters first
        self.search_time = episode_duration
        self.simulation_dt = 1.0 / simulation_frequency
        self.rl_dt = 1.0 / rl_frequency
        
        # 2. Extract paths from kwargs for the launcher
        scene_path = kwargs.get("scene_path")
        resources_path = kwargs.get("resources_path")
        
        # 3. Launch the simulator (Specific to this instance)
        if scene_path and resources_path:
            self.process = launch_stonefish_simulator(
                scene_path, resources_path, 
                observation_config_path, action_config_path, 
                port=(base_port + env_id),
                resolution=resolution,
                graphical=graphical,
            )
            # Give the simulator a moment to bind the socket
            time.sleep(2.0) 
        
        # 4. Initialize parent (ZMQ connection)
        super().__init__(observation_config_path, action_config_path, env_id, base_port)

        # 5. Application specific init
        self.step_counter = 0
        self.target_threshold = 2
        self.goal_pose = np.array([-5.5, 0, 5.2])
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        
        

    def build_reset_command(self):
        """Build RESET command - specific to this application"""
        # ds_pos = [-3, -1, 5.0]
        # ds_rot = [0.0, 0.0, 0.0]

        girona_pos = [
            self.np_random.uniform(-6.0, 6.0),
            self.np_random.uniform(-3.0, 3.0),
            1.2
        ]
        girona_rot = [True,
            self.np_random.uniform(-np.pi, np.pi),
            0.0,
            0.0
        ]

        ds_pos = [
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(-1.0, 1.0),
            5.5
        ]
        ds_rot = [
            self.np_random.uniform(-np.pi, np.pi),
            0.0,
            0.0
        ]

        return [
            {"name": "girona500", "position": girona_pos, "rotation": girona_rot},
            {"name": "ds", "position": ds_pos, "rotation": ds_rot}
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
        self.last_action_applied = np.array(action, dtype=np.float32).flatten()
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Application logic
        if not self._auv_observed_ds():
            obs[0:4] = 0.0 # Efficiently zero out the slice
            additional_reward = -10.0        
        else:
            additional_reward = self.calculate_additional_reward()
            
        # Update termination/truncation
        terminated = terminated or self._is_terminated()
        truncated = truncated or self._is_truncated()
            
        
        # Goal check
        if additional_reward == 0:
            print(f"[Port {self.port}] Goal achieved!")
            terminated = True
        
        total_reward = reward + additional_reward
        info.update(self._get_additional_info())
        
        return obs, total_reward, terminated, truncated, info
    

    def calculate_additional_reward(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        robot_pos = self._get_observation_by_pattern("auv_pose")
        ball_pos = self._get_observation_by_pattern("ds_pose")
        # collision_flag = self._get_observation_by_pattern("collision", default=0.0)

        # print(f"[DEBUG] robot_pos: {robot_pos}, collision_flag: {collision_flag}")
        # Calculate distances
        dist_to_target = self._distance_to_target(robot_pos[:3])
        
        reward = 0.0
        # print(f"[DEBUG] dist_to_target: {dist_to_target}, dist_to_goal: {dist_to_goal}, collision_flag: {collision_flag}")
        if dist_to_target > 0.1: # this distance is the ellips summation of its radia 
            reward = dist_to_target*-1 # this value makes the weight moving the ball equivalent to reaching the ball
        else:
            return 0 # AUV Achieved the Goal

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
        return self.step_counter * self.rl_dt >= self.search_time
    
    def _auv_observed_ds(self):
        """Check if AUV has observed the docking station"""
        ds_pos = self._get_observation_by_pattern("ds_pose")[:2]
        auv_pos = self._get_observation_by_pattern("auv_pose")[:2]
        distance = np.linalg.norm(ds_pos - auv_pos)
        # Consider observed if within 10 meters
        return distance < 2.0

    def _get_additional_info(self):
        """Additional info for this application"""
        robot_pos = self._get_observation_by_pattern("auv_pose")
        return {
            "distance_to_target": self._distance_to_target(robot_pos[:3]),
            "ds_position": self._get_observation_by_pattern("ds_pose").tolist(),
            "ds_observed": self._auv_observed_ds(),
            "step": self.step_counter
        }