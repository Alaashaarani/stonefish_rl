import zmq
import yaml
import gymnasium as gym
import numpy as np
import os 
import subprocess  
from stonefish_rl.utils.utils import resolve_path
      

class EnvStonefishRLParallel(gym.Env):
    """Base Gymnasium wrapper around one Stonefish simulator process.

    Child environments are responsible for task-specific behavior. Before
    calling this base constructor, a child class must set:

    - `self.state_path`: YAML file that defines simulator observations.
    - `self.act_path`: YAML file that defines actuator commands.
    - `self.tcm`: optional action transform matrix, or `None`.
    - `self.observe_actions`: whether to append the current action to observations.
    - `self.history_length`: number of previous observations to concatenate.
    - `self.process`: the `subprocess.Popen` returned by `launch_stonefish_simulator`.

    To create a new task environment, subclass this class and usually override:

    - `build_reset_command`: choose robot/object reset poses and current.
    - `reset`: clear task-specific counters/history and return the first observation.
    - `step`: shape observations, compute task reward, and return Gymnasium's
      `(obs, reward, terminated, truncated, info)` tuple. The base `step`
      method is a low-level command helper that still returns the older
      `(obs, reward, done, info)` tuple, so new tasks should normally override
      it as the docking and football environments do.
    - `_calculate_reward`, `_is_done`, and `_get_info` only if you deliberately
      use the base `step` implementation directly.
    """

    def __init__(self,
                 action_size, 
                 env_id=0, 
                 base_port=5555):
        super().__init__()

        """ Defined Values in the child CLASS: 
        self.state_path 
        self.act_path 
        self.tcm 
        self.observation_history
        self.observe_actions
        """ 

        self.env_id = env_id
        self.port = base_port + env_id
        self.ip = f"tcp://localhost:{self.port}"
        
        print(f"[ENV {self.env_id}] Connecting to {self.ip}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # Critical: Increase timeout for the first connection as Stonefish takes time to load    
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.ip)

        # self.socket.setsockopt(zmq.IMMEDIATE, 1)

        self.socket.connect(self.ip)
        
        # Load configurations (paths are defined in the Child Class)
        self.state_config = self._load_config(self.state_path)
        self.action_config = self._load_config(self.act_path)
        
        self.state_names = self._get_state_names()
        self.action_names = self._get_action_names()
        

        if isinstance(action_size,int):
            self.action_size = action_size
        else : 
            self.action_size = len(self.action_names)

        
        self.state_size = len(self.state_names)
        # adding action to observations
        if self.observe_actions:
            self.obs_size = self.state_size + self.action_size
        else: 
            self.obs_size = self.state_size
    
        # increasing the total size to save history observations
        self.total_obs_size = self.obs_size * (1+self.history_length)
        
        # Define spaces
        action_low, action_high = self._get_action_bounds()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=(self.action_size,), dtype=np.float32
        )
        
        self.state = np.zeros(self.total_obs_size, dtype=np.float32)
        self.msg_count = 0
        self.runtime_params = {}
        self.process = None # Initialize as None; child class will assign the Popen object
        
    def _load_config(self, config_path):
        """Load YAML configuration file."""
        config_path = resolve_path(config_path)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data or {}
        except FileNotFoundError:
            print(f"[WARNING] Config file {config_path} not found, using empty config")
            return {}
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML config file {config_path}: {e}")
            return {}

    def _get_state_names(self):
        """Extract observation names from observation config"""
        names = []
        try:
            specs = self.state_config.get("state_config", {}).get("specs", [])
            for spec in specs:
                names.append(spec.get("output_name", "unknown_observation"))
            print(f"[DEBUG] Added observation name: {names}")
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

    def _process_state_vector(self, msg):
        """Process YAML state vector from C++"""
        try:
            state_vector = yaml.safe_load(msg)
            if state_vector is None:
                state_vector = []
            if not isinstance(state_vector, list):
                raise ValueError(f"Expected YAML sequence for state vector, got {type(state_vector).__name__}")
            if len(state_vector) != self.state_size:
                print(f"[WARNING_here] State size mismatch: expected {self.state_size}, got {len(state_vector)}")

            self.state = np.array(state_vector, dtype=np.float32)

        except (yaml.YAMLError, ValueError, TypeError) as e:
            print(f"[ERROR] Failed to decode YAML state vector: {e}")
            self.state = np.array([], dtype=np.float32)

        return self.state

    def _process_action_vector(self, action):
        '''
        TCM is defined in the Child Class, because it is loaded from the config file 
        '''
        if isinstance(self.tcm,np.ndarray):
            return action@self.tcm
        else :
            return action

    def build_command(self, action_vector):
        """Build the `CMD:...;OBS:` string sent to Stonefish.

        New environments normally do not need to edit this. Change the action
        YAML when you need different actuator names, ranges, or command types.
        Override `_process_action_vector` or set `self.tcm` when the RL action
        space differs from the simulator actuator command space.
        """
        # if len(action_vector) != self.action_size:
        #     print(f"[ERROR] Action vector size mismatch: expected {self.action_size}, got {len(action_vector)}")
        #     return "CMD:;OBS:"
        

        action_vector = self._process_action_vector(action_vector)

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
        self.socket.send_string(message)
        self.msg_count+=1        
        response = self.socket.recv_string()
        return response

    def reset_communication(self):
        """
        Forcefully closes and re-initializes the ZMQ socket and context 
        to clear RAM and sync with the C++ reset.
        """
        # 1. Close existing socket immediately
        if hasattr(self, 'socket') and self.socket:
            # LINGER 0 ensures messages in the OS buffer are dropped instantly
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
        
        # 2. Terminate and recreate context to ensure all C++ backend memory is freed
        if hasattr(self, 'context') and self.context:
            self.context.term()
        
        self.context = zmq.Context()
        
        # 3. Re-initialize the REQ socket
        self.socket = self.context.socket(zmq.REQ)
        
        # 4. Re-apply your specific options
        self.socket.setsockopt(zmq.RCVHWM, 2)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.SNDTIMEO, 30000)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        # 5. Reconnect
        self.socket.connect(self.ip)
        print(f"[ENV {self.env_id}] Python ZMQ Communication Reset Successful")

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

    def update_runtime_params(self, params):
        """Update optional parameters used by interactive tools.

        Training and normal evaluation do not call this method, so default YAML
        behavior stays unchanged. GUI/test tools can call it before `reset()` or
        between steps. Child environments may override this method to map known
        keys to task-specific attributes.
        """
        if not isinstance(params, dict):
            return self.runtime_params

        self.runtime_params.update(params)
        return self.runtime_params

    def get_runtime_debug_info(self):
        """Return lightweight state useful for GUI/debug displays."""
        return {
            "runtime_params": dict(self.runtime_params),
            "state_names": list(getattr(self, "state_names", [])),
            "action_names": list(getattr(self, "action_names", [])),
            "step": getattr(self, "step_count", getattr(self, "msg_count", 0)),
        }

    def reset(self, seed=None, options=None):
        """Reset the Stonefish simulation and return the raw simulator state.

        Child classes usually call `super().reset(...)`, then reshape/normalize
        observations and reset task-specific variables.
        """
        super().reset(seed=seed)
        # We must build a valid reset command. Child classes override 'build_reset_command'
        command_payload = self.build_reset_command() if hasattr(self, 'build_reset_command') else {}
        reset_payload = yaml.safe_dump(command_payload, default_flow_style=False, sort_keys=False)
        reset_msg = "RESET:\n" + reset_payload
        
        # try: 
        #     self.reset_communication() 
        # except Exception as e:
        #     print(f"[ENV {self.env_id}] Failed to reset communication: {e}")

        
        try:
            response = self.send_command(reset_msg)
            self._process_state_vector(response)
        except Exception as e:
            print(f"[ENV {self.env_id}] Reset timeout/failure: {e}")
            # Optional: Add logic here to re-launch the specific simulator instance
        
        self.msg_count = 0
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
            # This loop is used when we are using force vector. If we are using Setpoints,
            # this function is not needed. 
            while len(lows)<self.action_size:
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
            self._process_state_vector(msg)
            
            reward = self._calculate_reward()
            done = self._is_done()
            info = self._get_info()
            
            return self.state, reward, done, info
            
        except Exception as e:
            print(f"[ERROR] Step failed: {e}")
            return self.state, 0.0, False, {}

    def _calculate_reward(self):
        """Calculate reward when a child class uses the base `step` method."""
        return 0.0

    def _is_done(self):
        """Return task termination when a child class uses the base `step` method."""
        return False

    def _get_info(self):
        """Return diagnostic info when a child class uses the base `step` method."""
        return {}

    def print_state(self):
        """Print current state with names"""
        print(f"[DEBUG] State ({len(self.state)} elements):")
        for i, (name, value) in enumerate(zip(self.state_names, self.state)):
            print(f"  [{i}] {name}: {value}")


# this function calls the cpp launcher of stonefish located in main.cpp 
def launch_stonefish_simulator(rank, config):
    """Launch one Stonefish process for a vectorized environment slot.

    New environments can reuse this launcher as long as their YAML config has
    the same `env` and `sim` path keys. The scene path can point to any Stonefish
    scenario, for example the docking scenario or the football scenario.
    """
    # Path to the Stonefish executable
    stonefish_exe = resolve_path("build/StonefishRL")
    num_instances = config["env"]["instances"]
    state_config_path= resolve_path(config["env"]["state_config"])
    action_config_path= resolve_path(config["env"]["action_config"])
    port=rank + config["env"]["base_port"]
    rl_freq = config["env"]["rl_observation_freq"]
    print(f"[INFO] Launching Stonefish on Port {port}...")

    scene_path= resolve_path(config["sim"]["scene_path"])
    resources_path= resolve_path(config["sim"]["resources_path"])
    # real_time= config["sim"]["realtime"] 
    resolution=config["sim"]["resolution"]
    graphical=config["sim"]["graphical_interface"]
    
    
    # Run the scene
    # Note: We pass the port as an additional command line argument to the C++ executable
    print(f"[INFO] Executing Stonefish on Port {port} with the scene: {scene_path}")
    
    # We use a process group (start_new_session) so we can kill this specific tree later
    if rank == 0 or rank == num_instances+1 : 
        pass
    else: 
        graphical = False

    vector = [stonefish_exe, 
        scene_path, 
        resources_path, 
        state_config_path, 
        action_config_path, 
        str(port),
        str(resolution),
        str(graphical),
        str(rl_freq)
        ]
    
        
    stonefish_proc = subprocess.Popen(vector,
        start_new_session=True )
    
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
        
