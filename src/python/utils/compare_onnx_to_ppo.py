import numpy as np
import openvino as ov
from stable_baselines3 import PPO,SAC


ppo_model_path = "./PPO_H2_A1_15M_20260515_163803.zip"
onnx_model_path = "./PPO_H2_A1_15M_20260515_163803.onnx"

model = PPO.load(ppo_model_path, device="cpu")

core = ov.Core()
compiled_model = core.compile_model(onnx_model_path, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

obs_dim = model.observation_space.shape[0]

diffs = []

for i in range(100):
    obs = np.random.uniform(-1.0, 1.0, size=(obs_dim,)).astype(np.float32)

    action_ppo, _ = model.predict(obs, deterministic=True)
    action_ppo = np.asarray(action_ppo, dtype=np.float32).reshape(-1)

    result = compiled_model({input_layer: obs.reshape(1, obs_dim)})
    action_ov = np.asarray(result[output_layer], dtype=np.float32).reshape(-1)

    # Safety clip, should already be clipped in ONNX
    action_ov = np.clip(action_ov, model.action_space.low, model.action_space.high)

    diff = np.abs(action_ppo - action_ov)
    diffs.append(np.max(diff))

    if np.max(diff) > 1e-3:
        print("Large diff at test", i)
        print("obs:", obs)
        print("PPO:", action_ppo)
        print("OV: ", action_ov)
        print("diff:", action_ppo - action_ov)
        break

diffs = np.array(diffs)

print("Mean max diff:", np.mean(diffs))
print("Max diff:", np.max(diffs))
print("Std diff:", np.std(diffs))