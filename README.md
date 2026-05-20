# Stonefish RL

Reinforcement learning tools for docking experiments in the
[Stonefish](https://github.com/patrykcieslak/stonefish) simulator.

This repository contains a C++ Stonefish server and Python Gymnasium
environments. Python launches one or more Stonefish simulator processes,
communicates with them over ZeroMQ, and trains/evaluates Stable-Baselines3
agents.

## What Is Included

- `src/cpp/`: Stonefish launcher, command processing, state publishing, and ZMQ communication.
- `src/python/stonefish_rl/envs/`: base, docking, and football environments.
- `src/python/stonefish_rl/training/`: single-run and multi-run training tools.
- `src/python/stonefish_rl/evaluation/`: single-model and multi-model evaluation tools.
- `src/python/stonefish_rl/gui/`: interactive testing GUI and comparison-evaluation GUI.
- `src/python/*.py`: small compatibility entry points for the most common commands.
- `include/parameters/*.yaml`: editable run configurations.
- `include/observations/*.yaml`: observation and action definitions.

## Requirements

System dependencies:

- Stonefish `>= 1.5.0`
- CMake
- C++17 compiler
- `libzmq`
- `yaml-cpp`
- Python 3

Python dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install wandb
```

`wandb` is only needed when `log.enable_wandb: true` in the training YAML.
The GUIs use Python's built-in `tkinter` and the existing `matplotlib`
dependency. On some Linux installs, `tkinter` is packaged separately as
`python3-tk`.

## Build

From the repository root:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

This creates:

```text
build/StonefishRL
```

The Python environment expects that executable to exist before testing,
training, or evaluation.

## Paths In YAML Files

Configuration files use repository-relative paths. You do not need to write
full absolute paths like `/home/.../stonefish_rl/...`.

Use:

```yaml
state_config: "include/observations/ds_state_v2_config.yaml"
action_config: "include/observations/ds_action_config.yaml"
scene_path: "Resources/scenarios/girona_ds/girona1000_rl_docking_pool.scn"
resources_path: "./"
```

Model paths can also be repository-relative. New trained models are saved under
`models/`. Older models in `src/python/` can still be referenced explicitly.

## Main Config Files

`include/parameters/train_param.yaml`

Single training configuration. Important fields:

- `env.instances`: number of parallel Stonefish environments.
- `env.base_port`: first ZMQ port used by the simulators.
- `model.algorithm`: `PPO`, `SAC`, or `TD3`.
- `model.pretrained`: continue from an existing model when true.
- `model.model_path`: model to load when `model.pretrained` is true.
- `train.total_timesteps`: training duration.
- `train.warm_start_path`: prefix for saved model names, for example `PPO_`.
- `train.save_name`: main saved model name.
- `log.log_dir`: training logs directory. Default is `logs`.

`include/parameters/multi_train.yaml`

Batch training configuration. Values can be a scalar for all runs or a list
with one value per run. Important fields:

- `training_config_num`: number of training runs.
- `reward_weights`: one reward-weight vector or one vector per run.
- `max_iteration`: maps to `train.total_timesteps`.
- `model_type`: maps to `model.algorithm`.
- `pre_trained`: maps to `model.pretrained`.
- `pretrained`: maps to `model.model_path`.
- `output_agent_name`: maps to `train.save_name`.
- `warm_start_path`: maps to `train.warm_start_path`.

`include/parameters/evaluation_param.yaml`

Evaluation and model-comparison configuration.

`include/parameters/test_param.yaml`

Small environment test configuration for manual/random/controller actions.

`include/parameters/football_param.yaml`

Example football-task configuration. It uses:

- `Resources/scenarios/football/football_pool.scn`
- `include/observations/football_state_config.yaml`
- `src/python/stonefish_rl/envs/football_env.py`

## Creating A New Environment

The quickest path is to copy `src/python/stonefish_rl/envs/football_env.py` or
`src/python/stonefish_rl/envs/docking_env.py`, then edit the task-specific
methods.

In your new environment class, update:

- `build_reset_command`: choose which robot/object names are reset, their poses, and optional current.
- `reset`: clear counters, histories, and task state at the start of each episode.
- `step`: call `super().step(action)`, compute reward, set `terminated` and `truncated`, and return info.
- Reward helper functions such as `calculate_additional_reward` or your own task-specific equivalent.
- Any state lookup helper if your observation names differ from the example patterns.

You also need matching YAML files:

- A state config in `include/observations/` that lists the values Stonefish should return.
- An action config in `include/observations/` that maps RL outputs to Stonefish actuators.
- A parameter file in `include/parameters/` that points to your scene, state config, and action config.

For a new Stonefish scenario, either point `sim.scene_path` to an existing
scenario or add a wrapper scenario like `Resources/scenarios/football/football_pool.scn`.

If you want the testing GUI to select the new environment, register it in:

```text
src/python/stonefish_rl/envs/factory.py
```

For GUI-editable runtime values, implement or extend:

```python
update_runtime_params(self, params)
```

Training and evaluation do not call this method, so the normal YAML-driven
behavior stays unchanged.

## Run A Quick Environment Test

```bash
python3 src/python/test_docking_env.py
```

Or pass a config explicitly:

```bash
python3 src/python/test_docking_env.py include/parameters/test_param.yaml
```

Run the football example:

```bash
python3 src/python/test_football_env.py
```

Run the interactive testing GUI:

```bash
python3 src/python/test_gui.py
```

The testing GUI can start an environment, reset it, step once, resume/pause
continuous stepping, stop the simulator, change reward weights/current/start
poses, and plot multiple curves from reward, observation entries, action
entries, or `info`. Use `New` to remove all curves, `Add` to add the selected
curve, `Remove` to remove a selected curve, and `Clear Plot` to keep the curves
but clear their data. Action modes are `zero`, `random`, `manual`, and
`controller`; controller mode uses the Logitech helper from `stonefish_rl.utils`.
During `Resume`, the GUI automatically resets when an episode ends so the
simulation keeps running until you press `Pause` or `Stop`.

The football reward is:

```text
-(robot_ball_weight * distance(robot, ball)
  + ball_goal_weight * distance(ball, goal)) / rl_observation_freq
```

A success bonus is added when the ball is within `football.goal_tolerance` of
`football.goal_position`.

If a previous simulator did not shut down cleanly:

```bash
pkill -9 StonefishRL
```

## Train One Agent

Edit:

```text
include/parameters/train_param.yaml
```

Then run:

```bash
python3 src/python/train_docking.py
```

Or pass a config explicitly:

```bash
python3 src/python/train_docking.py include/parameters/train_param.yaml
```

Outputs:

- Final trained models: `models/`
- Monitor/evaluation logs: `logs/`
- TensorBoard logs: `logs/runs/`
- W&B local files: `logs/wandb/`
- W&B callback checkpoints: `logs/wandb_models/`

Saved model names are built from:

```text
train.warm_start_path + train.save_name + timestamp
```

Example:

```text
PPO_H2_A1_10M_default_weights_20260515_173700.zip
```

## Train Multiple Agents

Edit:

```text
include/parameters/multi_train.yaml
```

Check what will be generated without launching training:

```bash
python3 src/python/multi_train.py --dry-run
```

Run all trainings:

```bash
python3 src/python/multi_train.py
```

Per-run YAML files are generated under:

```text
include/parameters/generated_multi_train/
```

## Evaluate A Model

Edit:

```text
include/parameters/evaluation_param.yaml
```

Then run:

```bash
python3 src/python/evaluate_docking.py
```

For comparing multiple models:

```bash
python3 src/python/evaluate_docking_compare.py
```

Or use the comparison-evaluation GUI:

```bash
python3 src/python/compare_evaluation_gui.py
```

The comparison GUI lets you select several models, choose `PPO`, `SAC`, `TD3`,
or `ONNX` per model, edit the rest of the evaluation YAML, and launch the same
`evaluate_docking_compare` backend.

Comparison results are written to `evaluation_results/` unless you pass
`--output-dir`.

## Typical Workflow

1. Build the C++ simulator executable.
2. Confirm the environment launches with `test_docking_env.py`.
3. Tune `train_param.yaml` for one run.
4. Use `multi_train.yaml` when running several reward/model variants.
5. Evaluate with `evaluate_docking.py` or `evaluate_docking_compare.py`.
6. Keep trained models in `models/` and logs in `logs/`.

## Troubleshooting

`Address already in use`

A previous Stonefish process is still running:

```bash
pkill -9 StonefishRL
```

`Config file not found`

Use repository-relative paths in YAML files, for example:

```yaml
action_config: "include/observations/ds_action_config.yaml"
```

`Model file not found`

For new models, put or look under:

```text
models/
```

For older models, use an explicit repo-relative path:

```yaml
model_path: "src/python/OLD_MODEL_NAME.zip"
```

`W&B import error`

Install W&B or disable it:

```yaml
log:
  enable_wandb: false
```

## Project Structure

```text
stonefish_rl/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── Resources/
│   └── scenarios/
├── include/
│   ├── observations/
│   │   ├── ds_action_config.yaml
│   │   ├── ds_state_v2_config.yaml
│   │   └── football_state_config.yaml
│   └── parameters/
│       ├── evaluation_param.yaml
│       ├── football_param.yaml
│       ├── multi_train.yaml
│       ├── test_param.yaml
│       └── train_param.yaml
├── logs/
├── models/
├── src/
│   ├── cpp/
│   └── python/
│       ├── stonefish_rl/
│       │   ├── envs/
│       │   ├── evaluation/
│       │   ├── gui/
│       │   ├── scripts/
│       │   ├── training/
│       │   └── utils/
│       ├── train_docking.py
│       ├── evaluate_docking_compare.py
│       ├── test_gui.py
│       └── compare_evaluation_gui.py
└── build/
    └── StonefishRL
```
