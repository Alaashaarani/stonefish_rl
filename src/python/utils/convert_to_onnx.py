
import torch
from stable_baselines3 import PPO, SAC

# 1. Force the model to load onto the CPU
# map_location ensures even if it was saved on GPU, it loads to CPU
device = torch.device("cpu")
model = SAC.load("/home/cirs_alaa/repositories/stonefish_rl/src/python/logs/SAC_docking_20260218_192754.zip", device=device)

# 2. Extract the policy and ensure it's on CPU
onnx_policy = model.policy.to(device)

# 3. Create dummy input explicitly on the same device
shape = model.observation_space.shape
# Using *shape unpacks the tuple correctly
dummy_input = torch.randn(1, *shape).to(device)
output_name = "SAC_O14_A6.onnx" 
# 4. Export
torch.onnx.export(
    onnx_policy, 
    dummy_input, 
    output_name, 
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print("Export successful on CPU!")