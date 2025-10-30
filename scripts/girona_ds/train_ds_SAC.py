import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# from girona_ds.dsEnv import dsEnv 
from scripts.girona_ds.football_env import dsEnv # Goal version

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from core.launch_stonefish import launch_stonefish_simulator

launch_stonefish_simulator("Resources/girona_ds/scenarios/girona500_docking_sim_pool.scn")

# Create the training environment
env = dsEnv()


# Create the 'logs' folder if it doesn't already exist
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Callback to evaluate and save the best model
eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=5000,          
    deterministic=True,
    render=False
)

# Create the PPO model
model = SAC("MlpPolicy", env, verbose=1)
# model = SAC.load("/home/cirs_alaa/repositories/stonefish_rl/scripts/girona_ds/training_model/best_model.zip", env=env)  # Continue training from a pre-trained model
# Train the model
model.learn(total_timesteps=800_000, reset_num_timesteps=False, callback=eval_callback)

# Save the model
model.save("SAC_g500_final")
