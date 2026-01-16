import zmq
import json
import gymnasium as gym
import numpy as np
import os 
import subprocess  
      

class EnvStonefishRLParallel(gym.Env):

    def __init__(self, observation_config_path, action_config_path, 
                 env_id=0, base_port=5555):
        super().__init__()
        self.env_id = env_id
        self.port = base_port + env_id
        self.ip = f"tcp://localhost:{self.port}"
        
        print(f"[ENV {self.env_id}] Connecting to {self.ip}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # Critical: Increase timeout for the first connection as Stonefish takes time to load
        self.socket.setsockopt(zmq.RCVTIMEO, 30000) # 30s for the first reset
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.ip)
        
        # Load configurations
        self.observation_config = self._load_config(observation_config_path)
        self.action_config = self._load_config(action_config_path)
        
        self.observation_names = self._get_observation_names()
        self.action_names = self._get_action_names()
        
        self.observation_size = len(self.observation_names)
        self.action_size = len(self.action_names)
        
        # Define spaces
        action_low, action_high = self._get_action_bounds()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=(self.action_size,), dtype=np.float32
        )

        self.state = np.zeros(self.observation_size, dtype=np.float32)
        self.step_count = 0
        self.process = None # Initialize as None; child class will assign the Popen object
        
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
            print(f"[DEBUG] Action names: {names}")
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
            # print(f"[DEBUG] Processed {len(self.state)} observations")
            
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
        # print(f"[CONN] Sending command: {message}")
        self.socket.send_string(message)
        # print(f"[PYTHON] Sending step action for step {self.step_count}")
        self.step_count+=1        
        response = self.socket.recv_string()
        # print(f"[CONN] Response received: {len(response)} chars")
        # print("[EnvStonefishRL] Observation:", response)
        return response

    def close(self):
        """Close environment and ensure subprocess is killed"""
        print(f"[ENV {self.env_id}] Shutting down...")
        
        # 1. Try to tell C++ to exit gracefully via ZMQ
        try:
            # We send EXIT:; to match your prefix:payload format
            self.socket.send_string("EXIT:;")
            
            # Use a short timeout so we don't hang if the simulator crashed
            self.socket.setsockopt(zmq.RCVTIMEO, 2000) 
            response = self.socket.recv_string()
            print(f"[ENV {self.env_id}] Simulator responded: {response}")
        except Exception as e:
            print(f"[ENV {self.env_id}] Graceful exit handshake failed: {e}")
        finally:
            self.socket.close()
            self.context.term()

        # 2. Ensure the specific Stonefish process is terminated
        # Note: self.process is assigned in the child class (dsEnv)
        if hasattr(self, 'process') and self.process is not None:
            try:
                print(f"[ENV {self.env_id}] Terminating Stonefish process...")
                self.process.terminate()
                # Wait up to 3 seconds for it to exit
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print(f"[ENV {self.env_id}] Force killing Stonefish (SIGKILL)...")
                self.process.kill() 
            except Exception as e:
                print(f"[ENV {self.env_id}] Error during process cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup if close() wasn't called"""
        # This is a safety net for when the object is garbage collected
        try:
            if hasattr(self, 'socket') and not self.socket.closed:
                self.close()
        except:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # We must build a valid reset command. Child classes override 'build_reset_command'
        command_payload = self.build_reset_command() if hasattr(self, 'build_reset_command') else {}
        reset_msg = "RESET:" + json.dumps(command_payload) + ";"
        
        try:
            response = self.send_command(reset_msg)
            self._process_observation_vector(response)
        except Exception as e:
            print(f"[ENV {self.env_id}] Reset timeout/failure: {e}")
            # Optional: Add logic here to re-launch the specific simulator instance
        
        self.step_count = 0
        return self.state, {}

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


# this function calls the cpp launcher of stonefish located in main.cpp 
def launch_stonefish_simulator(scene_relative_path, 
                               resources_path, 
                               observation_config_path, 
                               action_config_path, 
                               port=5555,
                               resolution=300,
                               graphical=False):
    """
    Launch the Stonefish simulator with a specific port for multi-instance support.
    """
    # Path to the Stonefish executable
    stonefish_exe = os.path.join(global_path("build"), "StonefishRLTest")
    
    # Run the scene
    # Note: We pass the port as an additional command line argument to the C++ executable
    print(f"[INFO] Executing Stonefish on Port {port} with the scene: {scene_relative_path}")
    
    # We use a process group (start_new_session) so we can kill this specific tree later
    stonefish_proc = subprocess.Popen(
        [stonefish_exe, 
         scene_relative_path, 
         resources_path, 
         observation_config_path, 
         action_config_path, 
         str(port),
         str(resolution),
         str(graphical)
         ],
        start_new_session=True 
    )
    
    return stonefish_proc

def kill_stonefish_process(process):
    """
    Kills a specific stonefish process instance instead of all of them.
    """
    try:
        if process:
            process.terminate() # or process.kill()
            process.wait(timeout=2)
            print("[INFO] Stonefish process terminated.")
    except Exception as e:
        print(f"[ERROR] Could not kill Stonefish: {e}")
        
def global_path(relative_path):
    """Get absolute path from project root"""
    # Path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    return os.path.join(project_root, relative_path)
