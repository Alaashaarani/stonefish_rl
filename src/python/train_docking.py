import os
import subprocess
import sys
import datetime as dt
import time
import wandb
import yaml

from stable_baselines3 import SAC,PPO,TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from utils.utils import global_path
from docking_env import dsEnv
from wandb.integration.sb3 import WandbCallback

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

def linear_schedule(initial_lr, final_lr):
    def schedule(progress_remaining):
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule



if __name__ == "__main__":
    # --- CLEANUP OLD PROCESSES ---
    print("[INFO] Cleaning up old Stonefish instances...")
    try:
        # Silently kill any existing StonefishRLTest processes
        subprocess.run(["pkill", "-9", "StonefishRLTest"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

    if len(sys.argv) > 1:
        try:
            config= load_config(sys.argv[1])
        except Exception as e:
            print(f"Error loading config file: {e}, Add config file path as first argument")
            sys.exit(1)

    else:
        config = load_config(global_path("include/parameters/train_param.yaml"))

    # 1. Configuration
    log_dir = config["log"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
        
    num_instances = config["env"]["instances"]

    # 2. Create Parallel Environments
    # VecMonitor logs episode rewards/lengths for tensorboard
    train_env = SubprocVecEnv([make_env(i,config) for i in range(num_instances)])
    train_env = VecMonitor(train_env, log_dir)

    # 3. Create Evaluation Environment (Separate instance for testing)
    time.sleep(2)  # Ensure different ports
    if config["sim"]["evaluation_graphical_interface"]:
        config["sim"]["graphical_interface"] = True
    eval_env = SubprocVecEnv([make_env(num_instances+2,config)]) # Use the next available ID
    eval_env = VecMonitor(eval_env, log_dir)

    # 4. Callback to evaluate and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=config["train"]["eval_freq"], # Adjust frequency for parallel steps
        deterministic=True,
        render=False
    )

    # wandb initilization 
    if config["log"]["enable_wandb"]:
        name = config["model"]["algorithm"]+"_"+config["model"]["policy"]+"_"+ dt.datetime.now().strftime("%Y%m%d_%H%M")
        run = wandb.init(
                    id = name if config["log"]["run_name"]=="default" else config["log"]["run_name"],
                    project=config["log"]["project_name"],
                    tags = config["log"]["tags"],
                    sync_tensorboard=True,  # This automatically uploads your SB3 logs
                    monitor_gym=True,       # This upload videos of the robot
                    save_code=True, )
        
        callback = [eval_callback, WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2,    )]
    else: 
        callback = eval_callback

    learning_rate = linear_schedule(
        config["model"]["learning_rate_start"],
        config["model"]["learning_rate_end"]
    )
    # 5. IModel Initilization 
    # MlpPolicy is standard for vector/sensor observations
    if config["model"]["algorithm"]=="SAC": 
        if config["model"]["pretrained"]: 
            model = SAC.load(config["model"]["model_path"],
                            env=train_env, 
                            tensorboard_log=f"runs/{run.id}" if config["log"]["enable_wandb"] else log_dir,
                            weights_only=config["model"]["weights_only"]
                            )
        else:
            model = SAC(
                config["model"]["policy"], 
                train_env, 
                train_freq=(1, "step"), # Collect steps from each env before updating
                gradient_steps=config["model"]["gradient_steps"],       # Do  gradient updates 
                verbose=1, 
                learning_rate=learning_rate,
                tensorboard_log=f"runs/{run.id}" if config["log"]["enable_wandb"] else log_dir,
                buffer_size=config["model"]["buffer_size"], 
                learning_starts=config["model"]["learning_starts"],
                ent_coef = config["model"]["ent_coef"]
            )
    elif config["model"]["algorithm"]=="PPO":
        if config["model"]["pretrained"]: 
            model = PPO.load(config["model"]["model_path"],
                            env=train_env, 
                            tensorboard_log=f"runs/{run.id}" if config["log"]["enable_wandb"] else log_dir,
                            weights_only=config["model"]["weights_only"]
                            )
        else: 
            model = PPO(config["model"]["policy"],
                        env=train_env,
                        learning_rate= learning_rate, 
                        n_steps= config["model"]["n_steps"], 
                        batch_size= config["model"]["batch_size"], 
                        ent_coef= config["model"]["ent_coef"], 
                        clip_range= config["model"]["clip_range"],
                        tensorboard_log=f"runs/{run.id}" if config["log"]["enable_wandb"] else log_dir
                        )
        
    elif config["model"]["algorithm"]=="TD3":
        if config["model"]["pretrained"]: 
            model = TD3.load(config["model"]["model_path"],
                            env=train_env, 
                            tensorboard_log=f"runs/{run.id}" if config["log"]["enable_wandb"] else log_dir,
                             weights_only=config["model"]["weights_only"]
                            )
        else:
            model = TD3(config["model"]["policy"],
                        env=train_env,
                        learning_rate= learning_rate, 
                        buffer_size=config["model"]["buffer_size"], 
                        batch_size= config["model"]["batch_size"], 
                        policy_delay= config["model"]["policy_delay"],
                        tensorboard_log=f"runs/{run.id}" if config["log"]["enable_wandb"] else log_dir
                        )


    # 6. Train the model
    print(f"Starting training with {num_instances} instances...")  
    starting_time = time.time()
    try:
        model.learn(
            total_timesteps=config["train"]["total_timesteps"],
            reset_num_timesteps=config["train"]["reset_num_timesteps"], 
            callback= callback
                    )
   
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if config["train"]["save_path"] == "default": 
            name = config["model"]["algorithm"] + "_" + config["train"]["save_name"] + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        else: 
            name = config["train"]["save_path"]+"/"+config["model"]["algorithm"] + "_" + config["train"]["save_name"] + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 7. Final Save and Cleanup
        model.save(name)
        train_env.close()
        eval_env.close()

    end_time = time.time()
    elapsed_time = end_time - starting_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes.")
