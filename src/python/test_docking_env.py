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
    :param config: Yaml file contains all required parameters
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
    if config["controller"]["logitech"]:
        controller = LogitechController(deadzone=0.1)
    
    testing_steps = config["testing"]["episodes"]*config["env"]["episode_duration"]*config["env"]["rl_observation_freq"]
    for step in range(testing_steps):
        
        # controller
        if config["controller"]["logitech"]:
            actions = [controller.get_thruster_values() for _ in range(num_instances)]
        # random
        elif config["controller"]["random"]:
            actions = [envs.action_space.sample() for _ in range(num_instances)]
        # constant force vector
        else:
            actions = [config["controller"]["force_vector"] for _ in range(num_instances)]

        # Step all environments simultaneously
        obs, rewards, dones, infos = envs.step(actions)

        if step %  config["testing"]["step_per_print"] == 0:
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