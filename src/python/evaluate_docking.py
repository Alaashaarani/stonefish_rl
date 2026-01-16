import os
import sys
import numpy as np
from stable_baselines3 import SAC
from EnvStonefishRL import global_path
from docking_env import dsEnv

if __name__ == "__main__":
    # 1. Setup Paths
    obs_path = global_path("include/observations/ds_observation_config.json")
    act_path = global_path("include/observations/ds_action_config.json")
    scene_path = global_path("Resources/girona_ds/scenarios/girona500_docking_sim_pool.scn")
    res_path = global_path("./")
    
    # Path to your best model
    model_path = "/home/cirs_alaa/repositories/stonefish_rl/src/python/logs/best_model.zip"

    # 2. Initialize the Single Environment
    # We set env_id=0 and a specific port to ensure it doesn't 
    # conflict with any background training processes
    env = dsEnv(
        observation_config_path=obs_path,
        action_config_path=act_path,
        resolution=600,
        env_id=0,
        base_port=5555,
        scene_path=scene_path,
        resources_path=res_path,
        graphical=True, # only one instance
        episode_duration=120
    )

    # 3. Load the Model
    # We pass the env to the load function to ensure the action/obs spaces match
    model = SAC.load(model_path, env=env)
    print(f"Model loaded from: {model_path}")

    # 4. Evaluation Loop
    num_episodes = 5
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_counter = 0

        print(f"\n--- Starting Episode {ep + 1} ---")

        while not (done or truncated):
            # deterministic=True is crucial for evaluation!
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action*100)
            
            total_reward += reward
            step_counter += 1

            if step_counter % 50 == 0:
                print(f"Step: {step_counter} | Current Reward: {reward:.2f} | Total: {total_reward:.2f} \n action: {action}")


        result_str = "SUCCESS (Goal Reached)" if done else "TIMEOUT (Truncated)"
        print(f"Episode {ep + 1} finished: {result_str}")
        print(f"Total Reward: {total_reward:.2f} in {step_counter} steps.")

    # 5. Cleanup
    env.close()
    print("\nEvaluation complete. Simulator closed.")