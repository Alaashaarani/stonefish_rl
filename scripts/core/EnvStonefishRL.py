import zmq
import json
import gymnasium as gym
import numpy as np
from core.util.receiver import Receiver


class EnvStonefishRL(gym.Env):

    def __init__(self, ip="tcp://localhost:5555", observation_config_path="observation_config.json", action_config_path="action_config.json"):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(ip)
        self.receiver = Receiver()

        # Load configurations from JSON files
        self.observation_config = self._load_config(observation_config_path)
        self.action_config = self._load_config(action_config_path)
        
        # Extract observation and action info
        self.observation_names = self._get_observation_names()
        self.action_names = self._get_action_names()
        
        self.observation_size = len(self.observation_names)
        self.action_size = len(self.action_names)
        
        # Initialize state and spaces
        self.state = np.array([]) 
        self.observation_space = None
        self.action_space = None
        
        print(f"[ENV] Loaded: {self.observation_size} observations, {self.action_size} actions")

    def _load_config(self, config_path):
        """Load JSON configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] Config file {config_path} not found, using empty config")
            return {}
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse {config_path}: {e}")
            return {}

    def _get_observation_names(self):
        """Extract observation names from observation config"""
        names = []
        try:
            specs = self.observation_config.get("observation_config", {}).get("specs", [])
            for spec in specs:
                names.append(spec.get("output_name", "unknown_observation"))
            return names
        except Exception as e:
            print(f"[ERROR] Failed to parse observation names: {e}")
            return []

    def _get_action_names(self):
        """Extract action names from action config"""
        names = []
        try:
            specs = self.action_config.get("action_config", {}).get("specs", [])
            for spec in specs:
                names.append(spec.get("output_name", "unknown_action"))
            return names
        except Exception as e:
            print(f"[ERROR] Failed to parse action names: {e}")
            return []

    def _process_observation_vector(self, msg):
        """Process observation vector from C++"""
        try:
            obs_vector = json.loads(msg)
            if len(obs_vector) != self.observation_size:
                print(f"[WARNING] Observation size mismatch: expected {self.observation_size}, got {len(obs_vector)}")
            
            self.state = np.array(obs_vector, dtype=np.float32)
            print(f"[DEBUG] Processed {len(self.state)} observations")
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to decode observation vector: {e}")
            self.state = np.array([], dtype=np.float32)
        
        return self.state

    def build_command(self, action_vector):
        """Build CMD string from action vector using action config"""
        if len(action_vector) != self.action_size:
            print(f"[ERROR] Action vector size mismatch: expected {self.action_size}, got {len(action_vector)}")
            return "CMD:;OBS:"
        
        parts = []
        try:
            specs = self.action_config.get("action_config", {}).get("specs", [])
            for i, (spec, action_value) in enumerate(zip(specs, action_vector)):
                actuator_name = spec.get("actuator_name", f"actuator_{i}")
                action_type = spec.get("action_type", "setpoint")
                parts.append(f"{actuator_name}:{action_type}:{action_value}")
                
            return "CMD:" + ";".join(parts) + ";OBS:"
            
        except Exception as e:
            print(f"[ERROR] Failed to build command: {e}")
            return "CMD:;OBS:"

    def send_command(self, message):
        """Send command to StonefishRL simulator"""
        print(f"[CONN] Sending command: {message}")
        self.socket.send_string(message)
        response = self.socket.recv_string()
        print(f"[CONN] Response received: {len(response)} chars")
        return response

    def close(self):
        """Close environment"""
        _ = self.send_command("EXIT")
        self.socket.close()
        self.context.term()
        print("[INFO] SIMULATION ENDED.")

    def reset(self, seed=None, options=None):
        """Reset environment"""
        try:
            reset_command = "RESET:{}"
            response = self.send_command(reset_command)
            self._process_observation_vector(response)
            
            # Initialize spaces based on config
            if self.observation_space is None:
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.observation_size,), 
                    dtype=np.float32
                )
                
            if self.action_space is None:
                # Get action bounds from config or use defaults
                action_low, action_high = self._get_action_bounds()
                self.action_space = gym.spaces.Box(
                    low=action_low, 
                    high=action_high, 
                    shape=(self.action_size,), 
                    dtype=np.float32
                )
            
            super().reset(seed=seed)
            info = {}
            return self.state, info
            
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            self.state = np.array([], dtype=np.float32)
            info = {}
            return self.state, info

    def _get_action_bounds(self):
        """Get action bounds from config or use defaults"""
        try:
            # Try to get bounds from config
            specs = self.action_config.get("action_config", {}).get("specs", [])
            lows = []
            highs = []
            
            for spec in specs:
                low = spec.get("min_value", -1.0)
                high = spec.get("max_value", 1.0)
                lows.append(low)
                highs.append(high)
                
            return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)
            
        except Exception as e:
            print(f"[WARNING] Failed to get action bounds from config, using defaults: {e}")
            return np.full((self.action_size,), -1.0), np.full((self.action_size,), 1.0)

    def step(self, action):
        """Execute one environment step"""
        try:
            message = self.build_command(action)
            msg = self.send_command(message)
            self._process_observation_vector(msg)
            
            reward = self._calculate_reward()
            done = self._is_done()
            info = self._get_info()
            
            return self.state, reward, done, False, info
            
        except Exception as e:
            print(f"[ERROR] Step failed: {e}")
            return self.state, 0.0, True, False, {}

    def _calculate_reward(self):
        """Calculate reward - to be overridden by child classes"""
        return 0.0

    def _is_done(self):
        """Check if done - to be overridden by child classes"""
        return False

    def _get_info(self):
        """Get additional info - to be overridden by child classes"""
        return {}

    def print_observation(self):
        """Print current observation with names"""
        print(f"[DEBUG] Observation ({len(self.state)} elements):")
        for i, (name, value) in enumerate(zip(self.observation_names, self.state)):
            print(f"  [{i}] {name}: {value}")