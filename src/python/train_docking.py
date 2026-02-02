import os
import subprocess
import sys
import datetime as dt
import time
import wandb

from stable_baselines3 import SAC,PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from EnvStonefishRL import global_path
from docking_env import dsEnv
from wandb.integration.sb3 import WandbCallback

def make_env(rank, 
             graphical=True,
             reso=400, 
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
            episode_duration=100,
            env_id=rank,               # Unique ID: 0, 1, etc.
            base_port=5555,            # Port will be 5555 + rank
            scene_path=scene_path,
            resources_path=res_path,
            graphical=graphical # if true, make sure to have low number of instances 
        )
        return env
    return _init

if __name__ == "__main__":
    # --- CLEANUP OLD PROCESSES ---
    print("[INFO] Cleaning up old Stonefish instances...")
    try:
        # Silently kill any existing StonefishRLTest processes
        subprocess.run(["pkill", "-9", "StonefishRLTest"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python train_docking.py [enable_graphical] [num_envs] [windows_resolution] ")
            sys.exit(0)
        elif sys.argv[1].lower() in ['true', '1', 'yes','false', '0', 'no']:
            enable_graphical = sys.argv[1].lower() in ['true', '1', 'yes']
            num_envs = 2  # default
            windows_resolution = 400  # default

        if len(sys.argv) == 3:
            if sys.argv[1].lower() in ['true', '1', 'yes','false', '0', 'no']:
                enable_graphical = not sys.argv[1].lower() in ['false', '0', 'no']
            if int(sys.argv[2]) > 20:
                print("WARNING: Number of instances is high.")
                num_envs = int(input("please re_enter a number of instances to confirm & press enter:"))
            else: 
                num_envs = int(sys.argv[2])
            
        
        if len(sys.argv) > 3:
            if sys.argv[1].lower() in ['true', '1', 'yes','false', '0', 'no']:
                enable_graphical = sys.argv[1].lower() in ['true', '1', 'yes']
            
            if int(sys.argv[2]) > 8 and enable_graphical:
                print("WARNING: Number of instances is too high for graphical mode.")
                num_envs = int(input("please re_enter a number of instances to confirm:"))
            else: 
                num_envs = int(sys.argv[2])
            
            windows_resolution = int(sys.argv[3])
    else:
        enable_graphical = False
        windows_resolution = 400
        num_envs = 2  # Set to 2 for two instances



    # 1. Configuration
    
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
    project="stonefish_docking",
    sync_tensorboard=True,  # This automatically uploads your SB3 logs
    monitor_gym=True,       # This upload videos of the robot
    save_code=True, )
    
    # 2. Create Parallel Environments
    # VecMonitor logs episode rewards/lengths for tensorboard
    train_env = SubprocVecEnv([make_env(i,graphical=enable_graphical,reso=windows_resolution) for i in range(num_envs)])
    train_env = VecMonitor(train_env, log_dir)

    # 3. Create Evaluation Environment (Separate instance for testing)
    # This ensures the best model is saved based on clean performance
    time.sleep(2)  # Ensure different ports
    eval_env = SubprocVecEnv([make_env(num_envs+2,reso=windows_resolution)]) # Use the next available ID
    eval_env = VecMonitor(eval_env, log_dir)

    # 4. Callback to evaluate and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000, # Adjust frequency for parallel steps
        deterministic=True,
        render=False
    )

    # 5. Initialize SAC
    # MlpPolicy is standard for vector/sensor observations
    model = SAC(
        "MlpPolicy", 
        train_env, 
        train_freq=(1, "step"), # Collect steps from each env before updating
        gradient_steps=1,       # Do  gradient updates 
        verbose=1, 
        # tensorboard_log=log_dir,
        tensorboard_log=f"runs/{run.id}",
        buffer_size=100000, 
        learning_starts=3000
    )

    # model = PPO(
    #     "MlpPolicy", 
    #     train_env, 
    #     verbose=1,
    #     tensorboard_log=f"runs/{run.id}"
    # )

    # Optional: Load previous model
    # model = SAC.load(best_model_path, env=train_env,tenserboard_log=f"runs/{run.id}")
    # model = SAC.load("sac_warm_started", env=train_env, tensorboard_log=f"runs/{run.id}")
    # model = SAC.load("SAC_run3", env=train_env, tensorboard_log=f"runs/{run.id}")

    starting_time = time.time()
    # 6. Train the model
    print(f"Starting training with {num_envs} instances...")
    try:
        model.learn(
            total_timesteps=1000_000,
            reset_num_timesteps=False, 
            callback=[eval_callback, WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
        
        )] )
   
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 7. Final Save and Cleanup
        model.save("SAC_run_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        train_env.close()
        eval_env.close()

    end_time = time.time()
    elapsed_time = end_time - starting_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes.")
