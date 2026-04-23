import os
import sys
import numpy as np
from stable_baselines3 import SAC,PPO,TD3
from utils.utils import global_path
from docking_env import dsEnv
import yaml 

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        try:
            config= load_config(sys.argv[1])
        except Exception as e:
            print(f"Error loading config file: {e}, Add config file path as first argument")
            sys.exit(1)

    else:
        config = load_config(global_path("include/parameters/evaluation_param.yaml"))

    # Path to your best model
    # model_path = "/home/cirs_alaa/repositories/stonefish_rl/src/python/SAC_girona_docking_final.zip"
    model_path = config["evaluate"]["model_path"]
    env = dsEnv(0, config)

    # 3. Load the Model
    if config["evaluate"]["algorithm"]=="SAC":
        model = SAC.load(model_path, env=env)
    elif config["evaluate"]["algorithm"]=="PPO":
        model = PPO.load(model_path, env=env)
    elif config["evaluate"]["algorithm"]=="TD3":
        model = TD3.load(model_path, env=env)
    elif config["evaluate"]["algorithm"]=="ONNX":
        import openvino as ov
        core = ov.Core()
        model = core.read_model(model_path)
        model = core.compile_model(model, "CPU") 

    print(f"Model loaded from: {model_path}")
    final_reward = 0
    # 4. Evaluation Loop
    num_episodes = config["evaluate"]["num_episodes"]
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_counter = 0

        print(f"\n--- Starting Episode {ep + 1} ---")

        while not (done or truncated):
            # deterministic=True is crucial for evaluation!
            
            # obs[3] += np.pi/2  # adjust heading observation because ds is oriented with half a pi
            if config["evaluate"]["algorithm"]=="ONNX":
                obs = obs.reshape(1,11)
                result = model(obs)
                action = result[0].reshape(6)
            else: 
                action, _ = model.predict(obs, deterministic=True)

            # downward actions 
            # action = np.array([0.0, 0.0, 0.0,-1.0,-1.0])

            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_counter += 1
            print(f"obs: {obs[3]:.2f}", end="\r")  
            if step_counter % config["evaluate"]["step_per_print"] == 0:
                print(f"Step: {step_counter} | Current Reward: {reward:.2f} | Total: {total_reward:.2f} \n action: {action}")

        final_reward += total_reward
        result_str = "SUCCESS (Goal Reached)" if done else "TIMEOUT (Truncated)"
        print(f"Episode {ep + 1} finished: {result_str}")
        print(f"Total Reward: {total_reward:.2f} in {step_counter} steps.")

    # 5. Cleanup
    env.close()
    print(f"\nEvaluation complete. Simulator closed. model: {model_path}, Average Reward: {final_reward/num_episodes:.2f}")