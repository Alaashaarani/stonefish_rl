import os, sys
import numpy as np
import time
import yaml
from docking_env import dsEnv
from EnvStonefishRL import global_path
from controller import LogitechController
from stable_baselines3.common.vec_env import SubprocVecEnv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def make_env(rank, config):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Paths to your configs
        

        env = dsEnv(rank, config)
        return env
    return _init

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            config= load_config(sys.argv[1])
        except Exception as e:
            print(f"Error loading config file: {e}, Add config file path as first argument")
            sys.exit(1)

    else:
        config = load_config(global_path("include/parameters/test_param.yaml"))

    
    num_instances = config["env"]["instances"]
    print(f"Starting {num_instances} instances...")

    # Create the vectorized environment
    # This spawns 'num_instances' separate python processes, each launching Stonefish
    envs = SubprocVecEnv([make_env(i,config) for i in range(num_instances)])

    print(f"\n--- {num_instances} Instances Ready ---")
    
    # Reset all environments
    obs = envs.reset()
    controller = LogitechController(deadzone=0.1)

    for step in range(10000):
        # Generate random actions for all environments
        # action shape will be (num_instances, action_size) randomly sampled
        # controller
        actions = [controller.get_thruster_values() for _ in range(num_instances)]
        
        #  random actions
        # actions = [envs.action_space.sample() for _ in range(num_instances)]

        # downward action (hard coded)
        # actions = [[0.0, 0.0, 0.0,-1.0,-1.0] for _ in range(num_instances)]
        
        # Circular motion
        # actions = [[0.7,0.1,0.0,-0.0,-0.0]for _ in range(num_instances)]
        
        # stable 
        # actions = [[0.0,0.0,0.0,-0.0,-0.0]for _ in range(num_instances)]

        # Step all environments simultaneously
        obs, rewards, dones, infos = envs.step(actions)

            
        if step % 100 == 0:
            print(f"Step {step} | Rewards: {rewards} | observations[0]: {obs[0]}")

    print("Testing finished. Closing environments...")
    envs.close()

'''
Troubleshooting "Address already in use"

If you see an error saying a port is already in use, 
it means a previous test didn't close properly. You can quickly clean them up with:

Bash:
pkill -9 StonefishRLTest
'''