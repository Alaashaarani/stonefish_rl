
import torch
from stable_baselines3 import PPO, SAC

# 1. Force the model to load onto the CPU
# map_location ensures even if it was saved on GPU, it loads to CPU
device = torch.device("cpu")
model = PPO.load("./PPO_O11_A6_20260301_164458", device=device)

# 2. Extract the policy and ensure it's on CPU
onnx_policy = model.policy.to(device)

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # This calls SB3's internal logic to get the deterministic action
        # it returns a tuple (action, latent_state), we only want the action [0]
        return self.policy._predict(observation, deterministic=True)
    
onnx_policy_wrapper = OnnxablePolicy(model.policy)

# 3. Create dummy input explicitly on the same device
shape = model.observation_space.shape
# Using *shape unpacks the tuple correctly
dummy_input = torch.randn(1, *shape).to(device)
output_name = "PPO_O11_A6.onnx" 
# 4. Export
torch.onnx.export(
    onnx_policy_wrapper, 
    dummy_input, 
    output_name, 
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print("Export successful on CPU!")