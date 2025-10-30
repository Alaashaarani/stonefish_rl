import os, sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from scripts.girona_ds.football_env import dsEnv
from stable_baselines3 import PPO, SAC
from core.launch_stonefish import launch_stonefish_simulator

launch_stonefish_simulator("Resources/girona_ds/scenarios/girona500_docking_sim_pool.scn")

# model = PPO.load("logs/ppo_best_model.zip")  
# model = SAC.load("/home/cirs_alaa/repositories/stonefish_rl/scripts/girona_ds/SAC_g500_final.zip")  
# model = SAC.load("/home/cirs_alaa/repositories/stonefish_rl/scripts/girona_ds/SAC_goal_final.zip")  
model = SAC.load("/home/cirs_alaa/repositories/stonefish_rl/scripts/girona_ds/final_models/SAC_g500_football_final.zip")

env = dsEnv()  
obs, _ = env.reset()


done = False
truncated = False
for i in range(10):
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        print(f"Reward: {reward}")
    

    if truncated:
        print("Truncated")
    else:
        print("Terminated")
    
    obs, _ = env.reset()
    done = False
    truncated = False