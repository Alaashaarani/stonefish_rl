import numpy as np
import json
from EnvStonefishRL import EnvStonefishRLParallel,launch_stonefish_simulator
import gymnasium as gym
import time
import gc

class dsEnv(EnvStonefishRLParallel):
    def __init__(self, rank, config,**kwargs):
        
        
        self.state_path = config["env"]["state_config"]
        self.act_path = config["env"]["action_config"]

        
        # 1. Store timing parameters first
        self.episode_duration = config["env"]["episode_duration"]  # seconds
        self.rl_observation_freq = config["env"]["rl_observation_freq"]  # Hz
        
        
        # 3. Launch the simulator (Specific to this instance)
        self.process = launch_stonefish_simulator(rank, config)            
        # Give the simulator a moment to bind the socket
        time.sleep(1.0) 
        

        """ Their should be a better way to define the action size """
        if  config["action"]["force_6Dof"]: 
        
            self.tcm = np.array(config["action"]["tcm"])
            
            action_size = self.tcm.shape[0]
        else: 
            self.tcm= None
            action_size = None

        self.observe_actions = config["env"]["observe_actions"]
        
        self.history_length = config["env"]["history_length"]
        if self.history_length <= 0: 
            self.history_length = 0


        # 4. 
        super().__init__(action_size, env_id=rank, base_port=config["env"]["base_port"])
        
        self.observation = np.array([], dtype=np.float32)
        self.observation_history = []

        # 5. Application specific init
        self.step_counter = 0
        self.n_calls=0
        self.check_freq = 2100
        self.goal_achieved = False
        self.start_distance_factor = 0.75
        self.collision_thres = 1.0 
        self.enable_currents = config["sim"]["current"]
        self.current_x = config["sim"]["current_value"][0]
        self.current_y = config["sim"]["current_value"][1]
        self.currnet_uniform = config["sim"]["current_uniform"]

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
            0.0+self.start_distance_factor*self.np_random.uniform(-np.pi, np.pi),
            0.0,
            0.0
        ]

        if self.enable_currents: 
            if self.currnet_uniform:
                current_vec = [np.random.uniform(-self.current_x,self.current_x),np.random.uniform(-self.current_y,self.current_y), 0.0]
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
        command = self.build_reset_command()
        reset_command = "RESET:" + json.dumps(command) + ";"
        
        # Use parent reset but send our specific reset command
        response = self.send_command(reset_command)

        
        self._process_state_vector(response)
        obs = self.get_sim_obser()
        
        self.step_counter = 0
        self.last_action_applied = np.zeros(self.action_size, dtype=np.float32)
        self.observation_history = []
        self.goal_achieved = False
        
        self.previous_acceleration = np.zeros(3)

        # with previous action 

        info = {}

        return obs, info

    def get_sim_obser(self):
        """Build observation: state vector + last action + zeros from history"""
        return self.concatenate_history()

    def step(self, action):
        """Execute step with cleaned logic"""
        self._on_step()
        self.step_counter += 1
        self.current_action = np.array(action, dtype=np.float32).flatten()
        
        # Getting the robot STATE from simulator. 
        obs, reward, terminated, truncated, info = super().step(action)

        if self.observe_actions:
            obs = np.concatenate((obs,self.current_action))
        # Application logic
        
        # Calculate the main rewards for the robot current state
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
        
        # reward comes from the parrent class and additional_reward comes from the child class
        total_reward = reward + additional_reward

        # Used for the smoothing reward
        self.last_action_applied = self.current_action

        info.update(self._get_additional_info())
        # adjusting state space to fit the observation requirements (adding offset,wrap, noise etc.)

        obs[2]-=1.25 # docking offset
        obs[3] += np.pi/2 
        error = np.linalg.norm(self._get_state_by_pattern("error")[:3])/6 # add noise to observations if ds is observed to prevent overfitting to perfect observations

        if not self._auv_observed_ds(): 
            obs[:3] += np.random.normal(0,error,3) # add noise to observations if ds is observed to prevent overfitting to perfect observations
        obs[:3] += np.random.normal(0,error/2,3)
        
        obs[3] = abs(np.clip(obs[3]+np.random.normal(0,0.1), -np.pi, np.pi))  # add noise to yaw error observation to prevent overfitting to perfect observations

        self.concatenate_history(obs)

        return np.round(self.observation,3), total_reward, terminated, truncated, {}
    
    def concatenate_history(self, obs=None): 
        """Concatenate observation history: 

            observation_history stores only the needed items for the observation,
            this can be improved by storing more history steps. Then select the needed
            items for the observation from the history.
        """

        if not isinstance(obs, np.ndarray) and len(self.observation) == 0:
            observation_history = np.zeros((self.history_length,self.obs_size))
            self.observation_history = list(observation_history)
            # return np.zeros(self.total_obs_size, dtype=np.float32)
             
        
        # print(f"obs: \n {obs}, \n history : \n {self.observation_history}")
        if isinstance(obs, np.ndarray):
            self.observation_history.append(obs)

        # print(f"observation history length: {len(self.observation_history)}")
        if len(self.observation_history) > self.history_length+1:
            self.observation_history.pop(0)
        
        self.observation = np.concatenate((self.observation_history), axis=0)
        # print(f"observation size after concatenation: {len(self.observation)}/{self.total_obs_size}")

        length = self.total_obs_size - len(self.observation)
        if length > 0:
            self.observation = np.concatenate((self.observation, np.zeros(length)))

        return self.observation

    def calculate_additional_reward(self):
        """Application-specific reward calculation"""
        # Extract relevant observations using their names
        error = self._get_state_by_pattern("error")
        imu_acc = self._get_state_by_pattern("imu_linear_acceleration")
        
        reward = 0.0 

        if not self._auv_observed_ds():
            reward += -1.0 # Penalty for losing sight of the docking station
    
        reward_dist = -abs(error[0])-abs(error[1])-(abs(error[2]))*0.5
        reward_yaw = np.exp(-2*abs(error[3]))-1
        action_difference = abs(self.current_action - self.last_action_applied).sum()
        
        smoothing_reward = -np.exp(action_difference)/(self.action_size*10)   # Exponential penalty for large action changes
            #     # print(f"[Port {self.port}] Action changed drastically! Difference: {action_difference:.2f}, Smoothing Reward: {smoothing_reward:.2f}")
            # smoothing_reward = 0 
            
        reward = reward_dist  + reward_yaw + smoothing_reward # this value makes the weight moving the ball equivalent to reaching the ball
        
        # Checking Collision
        if np.linalg.norm(self.previous_acceleration) != 0.0 :
            difference = np.linalg.norm(self.previous_acceleration - imu_acc)
            if difference > self.collision_thres : 
                reward -= 10 
                self.collision_thres+= self.collision_thres
                # print("collision Detected,  Reward: ", reward)
            elif self.collision_thres > 1.0: 
                self.collision_thres /= 2 

        # print(f"[Port {self.port}] reward_dist: {reward_dist:.2f}, reward_yaw: {reward_yaw:.2f}, collision: {difference}, action_diff: {action_difference:.2f}, smoothing_reward: {smoothing_reward:.2f}")
        
        if abs(error[2]) > 1.27 or abs(error[0]) > 0.15 or abs(error[1]) > 0.15 : 
            self.goal_achieved = False
        else:
            self.goal_achieved = True

        self.previous_acceleration = imu_acc
        reward /= self.rl_observation_freq    
        # print(f"[Port {self.port}] Distance to target: {error[0]:.2f} {error[1]:.2f} {error[2]:.2f} m , reward: {reward:.2f}", end='\r')
        # print(f"[DEBUG] Calculated additional reward: {reward}")    
        return reward

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
        collision_flag = self._get_state_by_pattern("collision", default=0.0)
        # Terminate if collision
        if collision_flag > 0.5:
            return True
            
        return False

    def _is_truncated(self):
        """Cleaned truncation logic""" 
        return self.step_counter/self.rl_observation_freq >= self.episode_duration
    
    def _auv_observed_ds(self):
        """Check if AUV has observed the docking station"""
        error = self._get_state_by_pattern("error")
        
        # check if AUV lies in DS cone field of view
        k = 0.64 # considering the cone size of (2,2,4.4) 
        if k*abs(error[2]) > abs(error[0]) and k*abs(error[2])> abs(error[1]):
            return True
        return False


    def _get_additional_info(self):
        """Additional info for this application"""
        # robot_pos = self._get_state_by_pattern("auv_pose")
        return {
            # "distance_to_target": self._distance_to_target(robot_pos[:3]),
            "error": self._get_state_by_pattern("error").tolist(),
            "ds_observed": self._auv_observed_ds(),
            "step": self.step_counter
        }