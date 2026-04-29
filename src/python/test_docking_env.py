import os, sys
import numpy as np
import time
import yaml
from docking_env import dsEnv
from utils.utils import LogitechController,RealTimePlotter,global_path
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

    plotter = RealTimePlotter(num_curves=1, max_entries=150)

    num_instances = config["env"]["instances"]
    print(f"Starting {num_instances} instances...")

    # Create the vectorized environment
    # This spawns 'num_instances' separate python processes, each launching Stonefish
    envs = SubprocVecEnv([make_env(i,config) for i in range(num_instances)])

    print(f"\n--- {num_instances} Instances Ready ---")

    # Reset all environments
    obs = envs.reset()
    if config["controller"]["logitech"]:
        controller = LogitechController(deadzone=0.1, use_forces= config["action"]["force_6Dof"])
    total_reward = 0
    #total testing steps
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
        obs, rewards, done, infos = envs.step(actions)
        # print(envs.step(actions))

        
        # I dont know why I cannot pass inf through the envs.step directly. 
        # infos = envs.get_attr('info')

        velocity = np.linalg.norm([obs[0][4], obs[0][5]])
        try: 
            value = infos[0]["smoothing_reward"] 
        except:
            value = 0.0
        plotter.update([value])
        # print("lenght : ",len(obs[0][-4:]),end="\n")
        # if terminate or truncated:
        #     print(f"\nEpisode finished at step {step}. Resetting environment... due to {'truncation' if truncated else 'terminate'}")
        #     obs = envs.reset()
        # total_reward += rewards
        # print(f"\r Step {step} | Reward: {total_reward}", end="")
        # if step %  config["testing"]["step_per_print"] == 0:
        #     print(f"\n Step {step} | Rewards: {rewards} | observations[0]: {obs[0]}, infos: {infos}", end="\n")
        #     print(f"step: {step}, actions: {actions}", end="\n")

    print("Testing finished. Closing environments...")
    envs.close()

'''
Troubleshooting "Address already in use"

If you see an error saying a port is already in use, 
it means a previous test didn't close properly. You can quickly clean them up with:

Bash:
pkill -9 StonefishRLTest
'''