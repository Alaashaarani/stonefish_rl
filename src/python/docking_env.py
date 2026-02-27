import numpy as np
import json
from EnvStonefishRL import EnvStonefishRLParallel,launch_stonefish_simulator
import gymnasium as gym
import time
import gc

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
        
        if  config["action"]["force_6Dof"]: 
        
            self.tcm = np.array(config["action"]["tcm"])
            
            action_size = self.tcm.shape[0]
        else: 
            self.tcm= None
            action_size = None
        

        # 4. 
        super().__init__(obs_path, act_path,action_size, env_id=rank, base_port=config["env"]["base_port"])
        


        # 5. Application specific init
        self.step_counter = 0
        self.n_calls=0
        self.check_freq = 2100
        self.goal_achieved = False
        self.start_distance_factor = 0.75
        self.collision_thres = 0.5 
        self.enable_currents = config["sim"]["current"]
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.current_action = np.zeros(self.action_size, dtype=np.float32)
        
    def _on_step(self) -> bool:
        self.n_calls+=1
        if self.n_calls % self.check_freq == 0:
            # print("*************** Calling GC *************")
            gc.collect()  # Force the scan
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
            else:          
                self.start_distance_factor = np.random.random()

        # method two, random factor each time 
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

        if self.enable_currents: 
            current_vec = [np.random.uniform(-0.1,0.1),np.random.uniform(-0.1,0.1), 0.0]
        else: 
            current_vec = [0.0,0.0,0.0]

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
        obs = self.get_sim_obser()
        # obs = np.concatenate((obs,self.last_action_applied))
        
        self.step_counter = 0
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.goal_achieved = False
        
        self.previous_acceleration = np.zeros(3)

        # with previous action 

        info = {}
        
        return obs, info

    def get_sim_obser(self):
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
        self._on_step()
        self.step_counter += 1
        self.current_action = np.array(action, dtype=np.float32).flatten()
        
        obs, reward, terminated, truncated, info = super().step(action)
        # obs = np.concatenate((obs,self.current_action))
        # Application logic
        
        additional_reward = self.calculate_additional_reward()
            
        # Update termination/truncation
        terminated = terminated or self._is_terminated()
        truncated = truncated or self._is_truncated()
            
        
        # Goal check
        if self.goal_achieved:
            print(f"[Port {self.port}] Goal achieved!")
            terminated = True
            additional_reward +=500
        
        if truncated:
            additional_reward -= 10.0  # Penalty for timeout
            
        total_reward = reward + additional_reward
        self.last_action_applied = self.current_action
        info.update(self._get_additional_info())
        # return np.zeros(22), 0, terminated, truncated, {}
        obs[2]-=1.25 # docking offset
        return np.round(obs,2), total_reward, terminated, truncated, {}
    

    def calculate_additional_reward(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        error = self._get_observation_by_pattern("error")
        imu_acc = self._get_observation_by_pattern("imu_linear_acceleration")
        
        
        if not self._auv_observed_ds():
            reward = -15.0 # Penalty for losing sight of the docking station
        else:
            reward_dist = -abs(error[0])-abs(error[1])-(abs(error[2])-1.25)*0.5
            reward_yaw = np.exp(-2*abs(error[3]))-1
            action_difference = abs(self.current_action - self.last_action_applied).sum()
            
            # if action_difference > 1:
            #     smoothing_reward = (-np.exp(action_difference - 1) + 1)/2  # Exponential penalty for large action changes
            #     # print(f"[Port {self.port}] Action changed drastically! Difference: {action_difference:.2f}, Smoothing Reward: {smoothing_reward:.2f}")
            # else:
            smoothing_reward = 0 
            
            reward = reward_dist  + reward_yaw + smoothing_reward # this value makes the weight moving the ball equivalent to reaching the ball
        
        # Checking Collision
        if np.linalg.norm(self.previous_acceleration) != 0.0 :
            difference = np.linalg.norm(self.previous_acceleration - imu_acc)
            if difference > self.collision_thres : 
                reward -= 10 
                self.collision_thres+= self.collision_thres
                # print("collision Detected,  Reward: ", reward)
            elif self.collision_thres > 0.5: 
                self.collision_thres /= 2 

        if abs(error[2]) > 1.27 or abs(error[0]) > 0.15 or abs(error[1]) > 0.15 : 
            self.goal_achieved = False
        else:
            self.goal_achieved = True

        self.previous_acceleration = imu_acc
        reward /= self.rl_observation_freq    
        # print(f"[Port {self.port}] Distance to target: {error[0]:.2f} {error[1]:.2f} {error[2]:.2f} m , reward: {reward:.2f}", end='\r')
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
        error = self._get_observation_by_pattern("error")
        
        # check if AUV lies in DS cone field of view
        k = 0.64 # considering the cone size of (2,2,4.4) 
        if k*abs(error[2]) > abs(error[0]) and k*abs(error[2])> abs(error[1]):
            return True
        return False


    def _get_additional_info(self):
        """Additional info for this application"""
        # robot_pos = self._get_observation_by_pattern("auv_pose")
        return {
            # "distance_to_target": self._distance_to_target(robot_pos[:3]),
            "error": self._get_observation_by_pattern("error").tolist(),
            "ds_observed": self._auv_observed_ds(),
            "step": self.step_counter
        }