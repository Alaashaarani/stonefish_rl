import os, sys
import numpy as np
import time
from docking_env import dsEnv
from EnvStonefishRL import global_path

from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank, 
             graphical=True,
             reso=800, 
             seed=0):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Paths to your configs
        obs_path = global_path("include/observations/ds_observation_config.json")
        act_path = global_path("include/observations/ds_action_config.json")
        scene_path = global_path("Resources/girona_ds/scenarios/girona500_docking_sim_pool.scn")
        res_path = global_path("./")

        env = dsEnv(
            observation_config_path=obs_path,
            action_config_path=act_path,
            resolution=reso,
            env_id=rank,               # Unique ID: 0, 1, etc.
            base_port=5555,            # Port will be 5555 + rank
            scene_path=scene_path,
            resources_path=res_path,
            graphical=graphical # if true, make sure to have low number of instances 
        )
        return env
    return _init

if __name__ == "__main__":
    # check if we recieve a command line argument for number of instances
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python test_docking_env.py [enable_graphical] [num_instances] [windows_resolution] ")
            sys.exit(0)
        elif sys.argv[1].lower() in ['true', '1', 'yes','false', '0', 'no']:
            enable_graphical = sys.argv[1].lower() in ['true', '1', 'yes']
            num_instances = 2  # default
            windows_resolution = 800  # default

        if len(sys.argv) == 3:
            if sys.argv[1].lower() in ['true', '1', 'yes','false', '0', 'no']:
                enable_graphical = not sys.argv[1].lower() in ['false', '0', 'no']
            if int(sys.argv[2]) > 20:
                print("WARNING: Number of instances is high.")
                num_instances = int(input("please re_enter a number of instances to confirm & press enter:"))
            else: 
                num_instances = int(sys.argv[2])
            
        
        if len(sys.argv) > 3:
            if sys.argv[1].lower() in ['true', '1', 'yes','false', '0', 'no']:
                enable_graphical = sys.argv[1].lower() in ['true', '1', 'yes']
            
            if int(sys.argv[2]) > 8 and enable_graphical:
                print("WARNING: Number of instances is too high for graphical mode.")
                num_instances = int(input("please re_enter a number of instances to confirm:"))
            else: 
                num_instances = int(sys.argv[2])
            
            windows_resolution = int(sys.argv[3])
    else:

        num_instances = 2  # Set to 2 for two instances
    
    # Create the vectorized environment
    # This spawns 'num_instances' separate python processes, each launching Stonefish
    envs = SubprocVecEnv([make_env(i,graphical=enable_graphical,reso=windows_resolution) for i in range(num_instances)])

    print(f"\n--- {num_instances} Instances Ready ---")
    
    # Reset all environments
    obs = envs.reset()

    for step in range(10000):
        # Generate random actions for all environments
        # action shape will be (num_instances, action_size)
        actions = [envs.action_space.sample() for _ in range(num_instances)]
        
        # Step all environments simultaneously
        obs, rewards, dones, infos = envs.step(actions)
        
        if step % 10 == 0:
            print(f"Step {step} | Rewards: {rewards}")

    print("Testing finished. Closing environments...")
    envs.close()

'''
Troubleshooting "Address already in use"

If you see an error saying a port is already in use, 
it means a previous test didn't close properly. You can quickly clean them up with:

Bash:
pkill -9 StonefishRLTest
'''