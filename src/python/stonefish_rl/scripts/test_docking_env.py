import os, sys
import numpy as np
import time
import yaml
from stonefish_rl.envs.docking_env import dsEnv
from stonefish_rl.utils.utils import LogitechController, resolve_path
from stable_baselines3.common.vec_env import SubprocVecEnv
import psutil

def print_process_tree_memory(tag=""):
    parent = psutil.Process(os.getpid())

    all_processes = [parent] + parent.children(recursive=True)

    print(f"\n[MEM TREE] {tag}")
    total = 0.0

    for p in all_processes:
        try:
            rss = p.memory_info().rss / 1024 / 1024
            total += rss
            print(f"pid={p.pid} name={p.name()} RSS={rss:.2f} MB")
        except psutil.NoSuchProcess:
            pass

    print(f"TOTAL RSS={total:.2f} MB\n")


def load_config(config_path):
    with open(resolve_path(config_path), 'r') as f:
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
        config = load_config(resolve_path("include/parameters/test_param.yaml"))

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
    total_reward = 0.0
    #total testing steps
    testing_steps = config["testing"]["episodes"]*config["env"]["episode_duration"]*config["env"]["rl_observation_freq"]
    for step in range(int(testing_steps)):

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
        obs, rewards, done, terminated = envs.step(actions)
        # print(envs.step(actions))
        print(f"[{step}] obs {obs}")
        info = envs.get_attr('info')

        velocity = np.linalg.norm([obs[0][4], obs[0][5]])
        try: 
            value2 = info[0]["reward_xy"] 
            value3 = info[0]["reward_z"]
            value4 = info[0]["reward_yaw"]
            value5 = info[0]["smoothing_reward"]
            value6 = info[0]["collision_reward"]
            value7 = info[0]["docking_accuracy_reward"]
        except:
            value2= value3= value4 = value5= value6 = 0.0

        value1 = rewards[0]
        if done: 
            print("done")
            total_reward = 0.0 
        else: 
            total_reward += rewards[0]

        # print("lenght : ",len(obs[0][-4:]),end="\n")
        # if terminate or truncated:
        #     print(f"\nEpisode finished at step {step}. Resetting environment... due to {'truncation' if truncated else 'terminate'}")
        #     obs = envs.reset()
        # total_reward += rewards
        # print(f"\r Step {step} | Reward: {total_reward}", end="")
        if step %  config["testing"]["step_per_print"] == 0:
            # print(f"\n Step {step} | Rewards: {rewards} | observations[0]: {obs[0]}, infos: {info}", end="\n")
            # print(f"\n Step {step} | Rewards: {rewards} ")
            # if rewards > 0: 
            #     break
            # print(f"step: {step}, actions: {actions}", end="\n")
            # print_process_tree_memory() 
            pass

    print("Testing finished. Closing environments...")
    envs.close()

'''
Troubleshooting "Address already in use"

If you see an error saying a port is already in use, 
it means a previous test didn't close properly. You can quickly clean them up with:

Bash:
pkill -9 StonefishRL
'''
