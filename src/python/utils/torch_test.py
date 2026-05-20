import numpy as np
import torch
from stable_baselines3 import PPO


device = torch.device("cpu")

model_path = "./PPO_H2_A1_0507_current_best.zip"
model = PPO.load(model_path, device=device)
model.policy.eval()


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy, action_low, action_high):
        super().__init__()
        self.policy = policy

        self.register_buffer(
            "action_low",
            torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high",
            torch.tensor(action_high, dtype=torch.float32)
        )

    def forward(self, observation):
        action = self.policy._predict(observation, deterministic=True)
        action = torch.clamp(action, self.action_low, self.action_high)
        return action


wrapper = OnnxablePolicy(
    model.policy,
    model.action_space.low,
    model.action_space.high
).to(device)

obs_dim = model.observation_space.shape[0]

for i in range(10):
    obs = np.random.uniform(-1.0, 1.0, size=(obs_dim,)).astype(np.float32)

    action_sb3, _ = model.predict(obs, deterministic=True)

    obs_torch = torch.tensor(obs.reshape(1, obs_dim), dtype=torch.float32)
    with torch.no_grad():
        action_wrapper = wrapper(obs_torch).cpu().numpy()[0]

    print("Test", i)
    print("SB3:    ", action_sb3)
    print("Wrapper:", action_wrapper)
    print("Diff:   ", action_sb3 - action_wrapper)
    print("Max diff:", np.max(np.abs(action_sb3 - action_wrapper)))
    print()