import numpy as np
import json
from core.EnvStonefishRL import EnvStonefishRL
import gymnasium as gym


class dsEnv(EnvStonefishRL):
    def __init__(self, observation_config_path, action_config_path, search_time, ip="tcp://localhost:5555" ): 
                 
        
        # Pass config paths to parent
        super().__init__( observation_config_path, action_config_path, ip )

        self.step_counter = 0
        self.target_threshold = 2
        self.search_time = search_time
        self.sim_stonefishRL = 0.001
        self.dt = 0.02 # Simulation approximite Time per frame 
        self.goal_pose = np.array([-5.5, 0, 5.2])
        
        # Last action will be appended to observation
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        
        # Update observation space to include last action
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.observation_size + self.action_size,), 
            dtype=np.float32
        )

    def build_reset_command(self):
        """Build RESET command - specific to this application"""
        ds_pos = [0, 0, 5.0]
        ds_rot = [0.0, 0.0, 0.0]

        girona_pos = [
            self.np_random.uniform(-6.0, 6.0),
            self.np_random.uniform(-2.7, 3.5),
            0.5
        ]
        girona_rot = [
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
        obs.extend(self.last_action_applied.tolist())
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute step"""
        self.step_counter += 1
        
        # Store action for next observation
        self.last_action_applied = np.array(action, dtype=np.float32).flatten()
        
        # Use parent's step method
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add application-specific termination conditions
        if not terminated:
            terminated = self._is_terminated()
        if not truncated:  
            truncated = self._is_truncated()
            
        # Add application-specific reward
        additional_reward = self.calculate_additional_reward()
        total_reward = reward + additional_reward
        
        info.update(self._get_additional_info())
        
        return obs, total_reward, terminated, truncated, info

    def calculate_additional_reward(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        robot_pos = self._get_observation_by_pattern("position")
        collision_flag = self._get_observation_by_pattern("collision", default=0.0)
        
        # Calculate distances
        dist_to_target = self._distance_to_target(robot_pos)
        dist_to_goal = self._distance_to_goal(robot_pos)
        
        reward = 0.0
        
        # Reward for being close to target
        if dist_to_target < self.target_threshold:
            reward += 10.0 * (1.0 - dist_to_target / self.target_threshold)
        
        # Penalty for collisions
        if collision_flag > 0.5:
            reward -= 5.0
            
        # Reward for reaching goal
        if dist_to_goal < 1.0:
            reward += 100.0
            
        return reward

    def _get_observation_by_pattern(self, pattern, default=0.0):
        """Get observation value by name pattern"""
        for i, name in enumerate(self.observation_names):
            if pattern in name and i < len(self.state):
                print(f"[DEBUG] Found observation '{name}' matching pattern '{pattern}': {self.state[i]}")
                return self.state[i]
        return default

    def _distance_to_target(self, robot_pos):
        """Distance to target (ds)"""
        target_pos = np.array([0.0, 0.0, 5.0])
        return np.linalg.norm(robot_pos - target_pos)

    def _distance_to_goal(self, robot_pos):
        """Distance to goal"""
        return np.linalg.norm(robot_pos - self.goal_pose)

    def _is_terminated(self):
        """Application-specific termination conditions"""
        robot_pos = self._get_observation_by_pattern("position")
        collision_flag = self._get_observation_by_pattern("collision", default=0.0)
        
        # Terminate if reached goal
        if self._distance_to_goal(robot_pos) < 1.0:
            return True
            
        # Terminate if collision
        if collision_flag > 0.5:
            return True
            
        return False

    def _is_truncated(self):
        """Application-specific truncation conditions"""
        # Truncate if exceeded time limit
        if self.step_counter * self.dt >= self.search_time:
            return True
        return False

    def _get_additional_info(self):
        """Additional info for this application"""
        robot_pos = self._get_observation_by_pattern("position")
        return {
            "distance_to_target": self._distance_to_target(robot_pos),
            "distance_to_goal": self._distance_to_goal(robot_pos),
            "step": self.step_counter
        }