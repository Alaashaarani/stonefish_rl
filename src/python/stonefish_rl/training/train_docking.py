import os
import subprocess
import sys
import datetime as dt
import time
import wandb
import yaml
from pathlib import Path

from stable_baselines3 import SAC,PPO,TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stonefish_rl.envs.docking_env import dsEnv
from stonefish_rl.utils.utils import resolve_model_path, resolve_path
from wandb.integration.sb3 import WandbCallback

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_config(config_path):
    with open(resolve_path(config_path), 'r') as f:
        return yaml.safe_load(f)


def resolve_repo_path(path_value, default_path=None):
    if path_value in (None, "", "default"):
        return Path(default_path)

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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

def env_loop(config,num_instances): 
    list = []
    
    for i in range(num_instances): 
        list.append(make_env(i,config))

    return list

if __name__ == "__main__":
    # --- CLEANUP OLD PROCESSES ---
    print("[INFO] Cleaning up old Stonefish instances...")
    try:
        # Silently kill any existing StonefishRL processes
        subprocess.run(["pkill", "-9", "StonefishRL"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

    if len(sys.argv) > 1:
        try:
            config= load_config(sys.argv[1])
        except Exception as e:
            print(f"Error loading config file: {e}, Add config file path as first argument")
            sys.exit(1)

    else:
        config = load_config(resolve_path("include/parameters/train_param.yaml"))

    # 1. Configuration
    log_dir = resolve_repo_path(config["log"].get("log_dir"), REPO_ROOT / "logs")
    models_dir = resolve_repo_path(config["train"].get("save_path"), REPO_ROOT / "models")
    wandb_dir = log_dir / "wandb"
    tensorboard_dir = log_dir / "runs"
    wandb_model_dir = log_dir / "wandb_models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(wandb_model_dir, exist_ok=True)
        
    num_instances = config["env"]["instances"]

    # 2. Create Parallel Environments
    # VecMonitor logs episode rewards/lengths for tensorboard
    
    train_env = SubprocVecEnv(env_loop(config,num_instances))

    train_env = VecMonitor(train_env, str(log_dir))

    # 3. Create Evaluation Environment (Separate instance for testing)
    time.sleep(2)  # Ensure different ports
    if config["sim"]["evaluation_graphical_interface"]:
        config["sim"]["graphical_interface"] = True
    eval_env = SubprocVecEnv([make_env(num_instances+1,config)]) # Use the next available ID
    eval_env = VecMonitor(eval_env, str(log_dir))

    # 4. Callback to evaluate and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
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
                    dir=str(wandb_dir),
                    sync_tensorboard=True,  # This automatically uploads your SB3 logs
                    monitor_gym=True,       # This upload videos of the robot
                    save_code=True, )
        tensorboard_log = str(tensorboard_dir / run.id)
        
        callback = [eval_callback, WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=str(wandb_model_dir / run.id),
                    verbose=2,    )]
    else: 
        run = None
        tensorboard_log = str(tensorboard_dir / dt.datetime.now().strftime("local_%Y%m%d_%H%M%S"))
        callback = eval_callback

    learning_rate = linear_schedule(
        config["model"]["learning_rate_start"],
        config["model"]["learning_rate_end"]
    )
    # 5. IModel Initilization 
    # MlpPolicy is standard for vector/sensor observations
    if config["model"]["algorithm"]=="SAC": 
        if config["model"]["pretrained"]: 
            model = SAC.load(resolve_model_path(config["model"]["model_path"]),
                            env=train_env, 
                            tensorboard_log=tensorboard_log,
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
                tensorboard_log=tensorboard_log,
                buffer_size=config["model"]["buffer_size"], 
                learning_starts=config["model"]["learning_starts"],
                ent_coef = config["model"]["ent_coef"]
            )
    elif config["model"]["algorithm"]=="PPO":
        if config["model"]["pretrained"]: 
            model = PPO.load(resolve_model_path(config["model"]["model_path"]),
                            env=train_env, 
                            tensorboard_log=tensorboard_log,
                            weights_only=config["model"]["weights_only"]
                            )
        else: 
            model = PPO(config["model"]["policy"],
                        env=train_env,
                        learning_rate= learning_rate, 
                        n_steps= config["model"]["n_steps"], 
                        batch_size= config["model"]["batch_size"], 
                        # ent_coef= config["model"]["ent_coef"], 
                        clip_range= config["model"]["clip_range"],
                        tensorboard_log=tensorboard_log
                        )
        
    elif config["model"]["algorithm"]=="TD3":
        if config["model"]["pretrained"]: 
            model = TD3.load(resolve_model_path(config["model"]["model_path"]),
                            env=train_env, 
                            tensorboard_log=tensorboard_log,
                             weights_only=config["model"]["weights_only"]
                            )
        else:
            model = TD3(config["model"]["policy"],
                        env=train_env,
                        learning_rate= learning_rate, 
                        buffer_size=config["model"]["buffer_size"], 
                        batch_size= config["model"]["batch_size"], 
                        policy_delay= config["model"]["policy_delay"],
                        tensorboard_log=tensorboard_log
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
        model_name_prefix = config["train"].get(
            "warm_start_path",
            config["model"]["algorithm"] + "_",
        )
        name = models_dir / (
            model_name_prefix
            + config["train"]["save_name"] + "_"
            + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        # 7. Final Save and Cleanup
        model.save(str(name))
        train_env.close()
        eval_env.close()
        if run is not None:
            wandb.finish()

    end_time = time.time()
    elapsed_time = end_time - starting_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes.")
