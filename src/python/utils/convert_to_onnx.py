
if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.

import torch
from stable_baselines3 import PPO, SAC


device = torch.device("cpu")

model_path = "./PPO_docking_20261006_193435.zip"
output_name = "PPO_docking_TD_20261006.onnx"

model = PPO.load(model_path, device=device)
model.policy.to(device)
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

if obs_dim != 35:
    raise ValueError(f"Expected obs_dim=35, but model expects {obs_dim}")

dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32).to(device)

torch.onnx.export(
    wrapper,
    dummy_input,
    output_name,
    opset_version=11,
    input_names=["input"],
    output_names=["action"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "action": {0: "batch_size"},
    },
)

print(f"Exported: {output_name}")
print("Observation dim:", obs_dim)
print("Action low:", model.action_space.low)
print("Action high:", model.action_space.high)