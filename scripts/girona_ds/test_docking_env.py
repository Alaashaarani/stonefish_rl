import os, sys
import numpy as np
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# from girona_ds.dsEnv import dsEnv
from girona_ds.docking_env import dsEnv # Goal version
from core.launch_stonefish import launch_stonefish_simulator, global_path

scene_path = global_path("Resources/girona_ds/scenarios/girona500_docking_sim_pool.scn") # concatinate the relative path with the global path of this repository
resources_path = global_path ("./")
observation_config_path = global_path("include/observations/ds_observation_config.json")
action_config_path = global_path("include/observations/ds_action_config.json")

launch_stonefish_simulator(scene_path, resources_path, observation_config_path , action_config_path) 


env = dsEnv(observation_config_path,
            action_config_path,
            ip="tcp://localhost:5555",
            search_time=120 # sec
            )

obs, info = env.reset()
terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    cont_values = np.array([100,100,0,-100,-100])/100  # Neutral command for the 5 thrusters, surge1,2; sway 3, heave 4,5

    # action = env.action_space.sample()  # Random vector for the 11 actuator floats
    action = (cont_values).tolist() # go down
    obs, reward, terminated, truncated, info = env.step(action)   
    
    print("\n Step: ", env.step_counter)
    # print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated, "| Truncated:", truncated)

    if terminated:
        print("GOAL ACHIEVED :) Resetting environment...\n")
        observation, info = env.reset()
        
    if truncated:
        print("TRUNCATED: 30 seconds exceeded :(")
        observation, info = env.reset()

    terminated = False
    truncated = False
