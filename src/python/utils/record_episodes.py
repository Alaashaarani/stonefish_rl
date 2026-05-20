# recorder_script.py
import os, sys
import numpy as np
import time
from stonefish_rl.envs.docking_env import dsEnv
from stonefish_rl.utils.utils import LogitechController, resolve_path
from stable_baselines3.common.vec_env import SubprocVecEnv
import time   
controller = LogitechController()
# ... initialize your env ...


if __name__ == "__main__":
    # check if we recieve a command line argument for number of instances
    
    enable_graphical = True
    windows_resolution = 800
    num_instances = 1  # Set to 2 for two instances


    obs_path = resolve_path("include/observations/ds_state_v2_config.yaml")
    act_path = resolve_path("include/observations/ds_action_config.yaml")
    scene_path = resolve_path("Resources/scenarios/girona_ds/girona1000_rl_docking_pool.scn")
    res_path = resolve_path("./")

    env = env = dsEnv(
            observation_config_path=obs_path,
            action_config_path=act_path,
            resolution=800,
            real_time = True,
            episode_duration=2000, # sec
            env_id=0,               # Unique ID: 0, 1, etc.
            base_port=5595,            # Port will be 5555 + rank
            scene_path=scene_path,
            resources_path=res_path,
            graphical=True # if true, make sure to have low number of instances 
        )


    expert_observations = []
    expert_actions = []

    num_episodes_to_record = 10

    for ep in range(num_episodes_to_record):
        obs = env.reset()
        done = False
        print(f"Recording Episode {ep+1}...")
        start_time = time.time()
        while not done:
            action = controller.get_thruster_values()
            next_obs, total_reward, terminated, truncated, info = env.step(action)
            # Record data every 0.1 seconds
            if time.time() - start_time >= 0.1:
                start_time = time.time()
                expert_observations.append(obs)
                expert_actions.append(action)
            obs = next_obs
            
            if terminated or truncated : break

    # Save as a single dataset
    np.savez("expert_docking_data.npz", 
            obs=np.array(expert_observations), 
            actions=np.array(expert_actions))
