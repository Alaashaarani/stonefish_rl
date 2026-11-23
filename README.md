# Stonefish RL

Reinforcement-learning tools for
[Stonefish](https://github.com/patrykcieslak/stonefish) simulator experiments.
The project contains a C++ Stonefish server plus Python Gymnasium environments
for training, testing, evaluating, and comparing Stable-Baselines3 agents.

The current examples focus on AUV docking and a simple football scenario.

## Requirements

System dependencies:

- Stonefish `>= 1.5.0`
- CMake
- C++17 compiler
- `libzmq`
- `yaml-cpp`
- Python 3

## Quick Start

Build the Stonefish executable:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install wandb
```

`wandb` is only required when `log.enable_wandb: true`. The GUIs use
`tkinter`; on some Linux systems you may need to install `python3-tk`.

Test that the simulator and environment launch:

```bash
python3 src/python/test_docking_env.py
```

Train one agent:

```bash
python3 src/python/train_docking.py
```

Run a multi-training sweep:

```bash
python3 src/python/multi_train.py
```

Evaluate and compare models:

```bash
python3 src/python/evaluate_docking_compare.py
```

Plot evaluation results:

```bash
python3 src/python/plot_evaluation_result.py evaluation_results --gui
```

## Project Layout

```text
stonefish_rl/
+-- Resources/scenarios/                 # Stonefish scenario files
+-- include/
|   +-- observations/                    # Observation/action YAML definitions
|   +-- parameters/                      # Train, test, evaluation YAML files
+-- logs/                                # Training logs, W&B local files, monitors
+-- models/                              # Final trained agents
+-- evaluation_results/                  # Per-model evaluation outputs and plots
+-- src/cpp/                             # Stonefish C++ server
+-- src/python/
    +-- stonefish_rl/
    |   +-- envs/                        # Base, docking, football environments
    |   +-- evaluation/                  # Evaluation, compare, plotting
    |   +-- gui/                         # Testing and compare GUIs
    |   +-- scripts/                     # Test entry points
    |   +-- training/                    # Single and multi-training
    |   +-- utils/                       # Shared utilities
    +-- train_docking.py                 # Compatibility entry point
    +-- multi_train.py
    +-- evaluate_docking_compare.py
    +-- plot_evaluation_result.py
    +-- test_gui.py
    +-- compare_evaluation_gui.py
```

## Repository-Relative Paths

YAML files should use paths relative to the repository root. You do not need
absolute paths such as `/home/user/repositories/stonefish_rl/...`.

Use this style:

```yaml
env:
  state_config: "include/observations/ds_state_v2_config.yaml"
  action_config: "include/observations/ds_action_config.yaml"

sim:
  scene_path: "Resources/scenarios/girona_ds/girona1000_rl_docking_pool.scn"
  resources_path: "./"
```

New trained models are saved in `models/`. Training logs and W&B files are kept
under `logs/`. Evaluation results are kept under `evaluation_results/`.

## Main YAML Files

`include/parameters/train_param.yaml`

Single-agent training configuration.

`include/parameters/multi_train.yaml`

Generates and runs many training configurations from one base training YAML.

`include/parameters/evaluation_param.yaml`

Single-model evaluation and multi-model comparison configuration.

`include/parameters/test_param.yaml`

Manual/random/controller environment testing configuration.

`include/parameters/football_param.yaml`

Example configuration for the football environment.

Generated files are written to:

```text
include/parameters/generated_multi_train/
include/parameters/generated_gui/
```

## Important Config Options

Environment options:

```yaml
env:
  state_config: "include/observations/ds_state_v2_config.yaml"
  action_config: "include/observations/ds_action_config.yaml"
  history_length: 2
  history_gap: 0.0
  observe_actions: true
  yaw_observation: "angle"   # "angle" or "cos_sin"
  instances: 50
  base_port: 5555
  episode_duration: 60
  rl_observation_freq: 3
```

`yaw_observation` controls the observation vector shape:

- `"angle"` keeps `error_yaw` as one scalar. Use it for old models trained with
  the original observation vector.
- `"cos_sin"` replaces the scalar yaw error with `cos(error_yaw)` and
  `sin(error_yaw)`. This adds one value to the base observation.

`history_length` also changes the observation vector size. When evaluating old
models, the evaluation config must match the model's trained observation shape.
The compare evaluator can auto-correct this for SB3 `.zip` models when there is
a unique matching option.

`history_gap` controls the time spacing, in seconds, between selected history
observations. Use `0.0` for consecutive observation steps.

Reward weights:

```yaml
reward:
  weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

For docking, the six weights are used for `xy`, `z`, `yaw`, docking accuracy,
smoothness, and collision terms. `multi_train.yaml` can override them per run.

Learning-rate options:

```yaml
model:
  algorithm: "PPO"
  learning_rate: auto
```

Valid forms:

```yaml
model:
  learning_rate: auto       # use the SB3 default for the selected algorithm
```

```yaml
model:
  learning_rate: 0.0003     # constant learning rate
```

```yaml
model:
  learning_rate_start: 0.001
  learning_rate_end: 0.00005
```

`learning_rate_start` and `learning_rate_end` create a linear schedule. Each
value can also be `auto`.

## Training One Agent

Edit:

```text
include/parameters/train_param.yaml
```

Run with the default config:

```bash
python3 src/python/train_docking.py
```

Or pass a config explicitly:

```bash
python3 src/python/train_docking.py include/parameters/train_param.yaml
```

Common training fields:

```yaml
model:
  algorithm: "PPO"       # PPO, SAC, or TD3
  pretrained: false
  model_path: "models/model_to_continue_from.zip"

train:
  total_timesteps: 10000000
  save_path: "default"   # "default" means stonefish_rl/models
  warm_start_path: "PPO_"
  save_name: "H2_A1_10M"
```

Final model names use:

```text
<warm_start_path><save_name>_<YYYYMMDD_HHMMSS>.zip
```

Example:

```text
models/PPO_H2_A1_10M_20260528_170455.zip
```

Training outputs:

```text
models/              # final agents
logs/                # monitor logs, TensorBoard, W&B local files
logs/runs/           # TensorBoard
logs/wandb/          # W&B local run data
logs/wandb_models/   # W&B callback checkpoints
```

## Multi-Training

Edit:

```text
include/parameters/multi_train.yaml
```

Preview generated configs without training:

```bash
python3 src/python/multi_train.py --dry-run
```

Run all trainings:

```bash
python3 src/python/multi_train.py
```

When multi-training starts, a small control GUI opens automatically. Press
`Stop Current Training` to request a graceful stop for the active training. The
current training is stopped cleanly and multi-training continues with the next
run.

For a headless/server run:

```bash
python3 src/python/multi_train.py --no-gui
```

For terminal control, press `Ctrl+C` once to request a graceful stop for the
current training. Press `Ctrl+C` again during shutdown to abort the whole
multi-training sequence.

Values in `multi_train.yaml` can be:

- a scalar, reused for all runs;
- a one-item list, reused for all runs;
- a list with one value per run.

Example:

```yaml
base_config: "train_param.yaml"
training_config_num: 3

reward_weights:
  - [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

max_iteration:
  - 1000000
  - 2000000
  - 5000000

model_type: "PPO"
pre_trained: true
pretrained:
  - "PPO_H2_A1_10M_20260525_182750.zip"

output_agent_name:
  - "H2_A1_1M_R111100"
  - "H2_A1_2M_R111100"
  - "H2_A1_5M_R111100"

warm_start_path: "PPO_"
yaw_observation: "cos_sin"
history_length: 2
history_gap: 0.0
```

The single `pretrained` entry above is reused for all three runs.

### Multi-Stage Training

Multi-stage training runs every agent through several stages. Stage 2 loads the
model saved by stage 1, stage 3 loads the model saved by stage 2, and so on.

Example:

```yaml
multistage:
  use: true
  stages:
    - name: stage1
      max_iteration: 100000
      sim:
        current: false
        enable_noise: false

    - name: stage2
      max_iteration: 100000
      sim:
        current: false
        enable_noise: true

    - name: stage3
      max_iteration: 500000
      sim:
        current: true
        enable_noise: true
```

Any parameter defined inside a stage overrides the base training YAML for that
stage. You can use nested YAML as above, or simple override rows supported by
`multi_train.py`, for example:

```yaml
- "sim,current,false"
- [sim, enable_noise, true]
- max_iteration: 100000
```

Intermediate stage models are saved without timestamps under:

```text
models/multistage_intermediate/<generated_timestamp>/
```

The final stage is saved in `models/` with the normal timestamped final name.
The timestamp is created when the final stage actually finishes, so repeated
runs with the same agent name create separate dated model files.

## Testing GUI

Run:

```bash
python3 src/python/test_gui.py
```

The testing GUI is for interactive environment debugging. It supports:

- selecting the environment and YAML config;
- reset, single-step, resume, pause, and stop;
- random, zero, manual, and controller action modes;
- editing important runtime parameters such as reward weights, current, and
  starting positions when the environment exposes them;
- plotting multiple curves at the same time.

Plot controls:

- `New`: reset the plot and remove all curves.
- `Add`: add the selected reward, observation, action, or info curve.
- `Remove`: remove the selected curve.
- `Clear Plot`: keep selected curves but clear their data.

Runtime plotting outside the GUI has been removed, so the testing GUI is the
single place for live plots.

## Football Environment

The football example lives in:

```text
src/python/stonefish_rl/envs/football_env.py
include/parameters/football_param.yaml
```

Run a quick random-action test:

```bash
python3 src/python/test_football_env.py
```

Limit the number of episodes:

```bash
python3 src/python/test_football_env.py --episodes 5
```

The football reward combines:

- Euclidean distance between robot and ball;
- Euclidean distance between ball and goal.

## Evaluation

Single-model evaluation:

```bash
python3 src/python/evaluate_docking.py
```

Multi-model comparison:

```bash
python3 src/python/evaluate_docking_compare.py
```

Or use the GUI:

```bash
python3 src/python/compare_evaluation_gui.py
```

The compare GUI can:

- load models from the YAML file;
- load all `.zip` and `.onnx` models from a directory;
- select or unselect all models;
- remove selected models;
- choose the algorithm/model type per row;
- evaluate only the checked rows;
- generate and run a temporary evaluation YAML.

Useful compare options:

```yaml
compare:
  compare: true
  num_episodes: 50
  seed: 0
  auto_correct_observation: true
  batch_size: 20

  algorithms: ["PPO", "SAC"]
  model_paths:
    - "models/PPO_H2_A1_5M.zip"
    - "models/SAC_H2_A1_3M.zip"

  yaw_observation:
    - "angle"
    - "cos_sin"

  history_length:
    - 2
    - 0

  history_gap:
    - 0.0
    - 0.2

  evaluation_tag:
    - ""
    - "3hz"
```

`evaluation_tag` lets you evaluate the same model under different settings
without replacing the previous result. For example, tag `3hz` saves results
under `model_name_3hz`.

`batch_size` controls large comparisons. If you select 60 models and
`batch_size: 20`, the evaluator runs models 1-20, then 21-40, then 41-60.

Evaluation seeding is shared across models. Episode `N` uses `seed + N`, so all
models see the same randomized starts, currents, and noise sequence.

### Evaluation Outputs

Compare evaluation writes one folder per evaluated model identity:

```text
evaluation_results/
+-- PPO_H2_A1_5M/
|   +-- summary.json
|   +-- episodes.csv
+-- PPO_H2_A1_5M_3hz/
|   +-- summary.json
|   +-- episodes.csv
+-- summary.csv
+-- summary.json
+-- episodes_all_models.csv
+-- current_evaluation_checklist.csv
+-- current_evaluation_checklist.json
+-- evaluation_errors.csv
+-- evaluation_errors.json
+-- top_models/
```

If a model evaluation already exists, the untagged result is replaced in its
model folder. If you want to keep two evaluations of the same model, use
`evaluation_tag`.

When a model fails, the compare evaluator records the error and continues with
the remaining models. Check:

```text
evaluation_results/evaluation_errors.csv
evaluation_results/current_evaluation_checklist.csv
```

At the end of a comparison, the evaluator creates top-10 tables:

```text
evaluation_results/top_models/top_10_success_rate.csv
evaluation_results/top_models/top_10_mean_reward.csv
evaluation_results/top_models/top_10_collision_performance.csv
evaluation_results/top_models/top_10_tables.json
```

## Plotting Evaluation Results

Interactive plot GUI:

```bash
python3 src/python/plot_evaluation_result.py evaluation_results --gui
```

Command-line plotting:

```bash
python3 src/python/plot_evaluation_result.py evaluation_results --output-dir evaluation_results/plots1
```

The plotter reads per-model folders first, so it can see all current model
evaluations in `evaluation_results/`, not only the root `summary.json`.

The plot GUI supports:

- loading an evaluation folder;
- selecting and unselecting all models;
- choosing which models to include;
- renaming model display names;
- editing figure titles;
- previewing figures before saving;
- choosing the output plot folder name, such as `plots1` or `plots2`.

All figures use the same selected model order, so success rate, reward, steps,
collision plots, and subfigures stay aligned.

## Reward Function Plots

Docking reward visualization utility:

```bash
python3 src/python/utils/plot_reward_function.py --show
```

Useful options:

```bash
python3 src/python/utils/plot_reward_function.py --reward-mode split --cut half --show
```

The script can save 4D reward plots where `x`, `y`, and `z` are position errors
and color represents reward.

## Creating A New Environment

Use `football_env.py` or `docking_env.py` as a starting point:

```text
src/python/stonefish_rl/envs/football_env.py
src/python/stonefish_rl/envs/docking_env.py
```

For a new environment, you usually need to edit or implement:

- `build_reset_command`: choose reset poses, object names, currents, and task
  initialization values sent to Stonefish.
- `reset`: clear counters, histories, cached noise, and task state at episode
  start.
- `step`: call the base environment step, compute reward, set `terminated` and
  `truncated`, and fill the `info` dictionary.
- Reward helper functions, for example `calculate_additional_reward`, or a new
  task-specific reward function.
- State lookup helpers if your observation names differ from the docking or
  football examples.
- `update_runtime_params(self, params)` if the testing GUI should change
  runtime values such as current, start pose, or reward weights.

You also need:

- a Stonefish scenario in `Resources/scenarios/`;
- a state config in `include/observations/`;
- an action config in `include/observations/`;
- a parameter YAML in `include/parameters/`.

To make the environment selectable by scripts and GUIs, register it in:

```text
src/python/stonefish_rl/envs/factory.py
```

Training and evaluation remain YAML-driven. GUI-only runtime edits are applied
through `update_runtime_params` and do not change the normal train/evaluate
workflow.

## Troubleshooting

`Address already in use`

A previous Stonefish process is still running:

```bash
pkill -9 StonefishRL
```

`Config file not found`

Use repository-relative paths:

```yaml
action_config: "include/observations/ds_action_config.yaml"
```

`Model file not found`

For new models, look in:

```text
models/
```

For older models, use an explicit repository-relative path:

```yaml
model_path: "src/python/OLD_MODEL_NAME.zip"
```

`Observation shape mismatch`

Check these fields:

```yaml
env:
  yaw_observation: "angle"   # or "cos_sin"
  history_length: 2
  history_gap: 0.0
  observe_actions: true
```

For compare evaluation, keep this enabled when evaluating many older/newer
models together:

```yaml
compare:
  auto_correct_observation: true
```

`W&B import error`

Install W&B or disable it:

```yaml
log:
  enable_wandb: false
```

`No active status file found` in the multi-train control GUI

Start multi-training with:

```bash
python3 src/python/multi_train.py
```

The control GUI is connected to the active multi-training run. Running the GUI
alone only works while multi-training has already created its control/status
files.
