
if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.

import os
import sys
import numpy as np
from stable_baselines3 import SAC,PPO,TD3
from library.docking_env import dsEnv
from library.env_factory import make_env_instance
from library.task_registry import supports_fixed_pid
from library.pid_docking import PIDDockingPolicy, is_pid
from library.common import resolve_model_path, resolve_path
import yaml 
from library.configuration import load_config as load_workflow_config, DEFAULT_CONFIG

def load_config(config_path):
    return load_workflow_config(config_path, "evaluation")


def evaluation_seed(config):
    seed_value = config.get("evaluate", {}).get("seed", 0)
    if seed_value is None:
        return None
    return int(seed_value)
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        try:
            config= load_config(sys.argv[1])
        except Exception as e:
            print(f"Error loading config file: {e}, Add config file path as first argument")
            sys.exit(1)

    else:
        config = load_config(DEFAULT_CONFIG)

    # Path to your best model
    # model_path = "src/python/SAC_girona_docking_final.zip"
    use_pid = is_pid(config["evaluate"]["algorithm"], config["evaluate"]["model_path"])
    model_path = "PID" if use_pid else resolve_model_path(config["evaluate"]["model_path"])
    if use_pid:
        config["evaluate"]["algorithm"] = "PID"
    env_type = config.get("env", {}).get("type", "docking")
    if use_pid and not supports_fixed_pid(config):
        raise ValueError(f"Fixed PID supports docking only; use a trained model for {env_type}.")
    env = make_env_instance(env_type, 0, config)

    # 3. Load the Model
    if use_pid:
        model = PIDDockingPolicy(env, config.get("pid", {}))
    elif config["evaluate"]["algorithm"]=="SAC":
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
    base_seed = evaluation_seed(config)
    for ep in range(num_episodes):
        episode_seed = None if base_seed is None else base_seed + ep
        obs, _ = env.reset(seed=episode_seed)
        if use_pid:
            model.reset()
        done = False
        truncated = False
        total_reward = 0
        step_counter = 0

        print(f"\n--- Starting Episode {ep + 1} | seed={episode_seed} ---")

        while not (done or truncated):
            # deterministic=True is crucial for evaluation!
            
            # obs[3] += np.pi/2  # adjust heading observation because ds is oriented with half a pi
            if config["evaluate"]["algorithm"]=="ONNX":
                obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                result = model(obs_batch)
                action = np.asarray(result[0], dtype=np.float32).reshape(-1)
                if action.size != env.action_size:
                    raise ValueError(
                        "ONNX model action size does not match the TCM: "
                        f"model returned {action.size}, TCM expects {env.action_size}."
                    )
            else: 
                action, _ = model.predict(obs, deterministic=True)

            # downward actions 
            # action = np.array([0.0, 0.0, 0.0,-1.0,-1.0])

            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_counter += 1
            # print(f"obs: {obs[3]:.2f}", end="\r")
            print(f"action: {action}")  
            if step_counter % config["evaluate"]["step_per_print"] == 0:
                print(f"Step: {step_counter} | Current Reward: {reward:.2f} | Total: {total_reward:.2f} \n action: {action}")

        final_reward += total_reward
        result_str = "SUCCESS (Goal Reached)" if done else "TIMEOUT (Truncated)"
        print(f"Episode {ep + 1} finished: {result_str}")
        print(f"Total Reward: {total_reward:.2f} in {step_counter} steps.")

    # 5. Cleanup
    env.close()
    print(f"\nEvaluation complete. Simulator closed. model: {model_path}, Average Reward: {final_reward/num_episodes:.2f}")
