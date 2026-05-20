

import numpy as np
import torch as th
import types
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data.types import Transitions
from stonefish_rl.envs.docking_env import dsEnv
from stonefish_rl.utils.utils import resolve_path


obs_path = resolve_path("include/observations/ds_state_v2_config.yaml")
act_path = resolve_path("include/observations/ds_action_config.yaml")


# 1. LOAD DATA (Handling your KeyError)
print("--- Loading Expert Data ---")
data = np.load("expert_docking_data.npz")
# Check keys to be safe
obs_key = 'obs' 
acts_key = 'actions' 



transitions = Transitions(
    obs=data[obs_key],
    acts=data[acts_key],
    infos=np.array([{} for _ in range(len(data[obs_key]))]), 
    next_obs=np.roll(data[obs_key], -1, axis=0), 
    dones=np.zeros(len(data[obs_key]), dtype=bool) 
)

# 2. INITIALIZE ENV
env = dsEnv(obs_path, act_path, graphical=False)

# 3. CREATE SURROGATE (The Brain we train)
print("--- Training Surrogate Policy (Compatible with BC) ---")
rng = np.random.default_rng()

# We use this policy because the BC trainer knows exactly how to handle it
custom_net_arch = [256, 256]

surrogate_policy = ActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: 0.001,
    net_arch=custom_net_arch  # <--- MUST MATCH SAC
).to("cuda" if th.cuda.is_available() else "cpu")

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    policy=surrogate_policy,  # <--- WE TRAIN THIS, NOT SAC
    rng=rng
)

bc_trainer.train(n_epochs=20)

# 4. THE TRANSPLANT (Move weights to SAC)
print("--- Transplanting Weights to SAC Actor ---")
model = SAC("MlpPolicy",
            env,
            policy_kwargs={"net_arch": custom_net_arch},
            verbose=1)

# Copying the MLPExtractor (Features) and Action Net (Output)
# This maps the "muscle memory" directly into the SAC actor
model.policy.actor.latent_pi.load_state_dict(
    surrogate_policy.mlp_extractor.policy_net.state_dict()
)
model.policy.actor.mu.load_state_dict(
    surrogate_policy.action_net.state_dict()
)

# 5. SAVE
model.save("sac_warm_started")
print("\n[SUCCESS] SAC model initialized with your movements.")
env.close()
