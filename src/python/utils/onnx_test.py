import openvino as ov
import numpy as np

# 1. Initialize OpenVINO Runtime Core
core = ov.Core()

# 2. Read the ONNX model
# This prepares the model but doesn't optimize it for hardware yet
model = core.read_model("SAC_O14_A6.onnx")

# 3. Compile the model for a specific device
# Options: "CPU", "GPU", "NPU", or "AUTO"
compiled_model = core.compile_model(model, "CPU")

# 4. Prepare your input (Observation from your RL environment)
# Ensure it matches the shape the model expects: (Batch_Size, Observation_Dim)
count=0
while count < 100 : 
    obs = np.random.randn(1, *model.inputs[0].shape[1:]).astype(np.float32)
    # 5. Run Inference

    # You can pass the input directly to the compiled model
    results = compiled_model(obs)
    # 6. Extract the Action
    # results is a dictionary-like object; index 0 gives the first output tensor
    action = results[0]

    print(f"Agent Action: {action}")
    count+=1