# Stonefish RL Configuration Reference

Episode domain randomization uses [`domain_randomization.yaml`](domain_randomization.yaml)
for part mass, compound transforms, and thruster limits. See the
[setup and reset behavior](../../docs/domain_randomization.md). Runtime mode keeps the simulator open and applies only changed values.

Experimental PID/PPO scalar blending has a separate configuration,
[`train_pid_rl_blend.yaml`](train_pid_rl_blend.yaml), and
[training/testing guide](../../docs/pid_rl_blend.md).

Use [stonefish_rl.yaml](stonefish_rl.yaml) for docking training, testing, and
evaluation. This reference describes the settings consumed by the repository's
current Python launchers, not every option available in Stable Baselines3 (SB3).
Adding an arbitrary SB3 option to YAML does not make the trainer use it.

## Contents

- [Loading and overrides](#loading-and-overrides)
- [Environment and observations](#environment-and-observations)
- [Simulator settings](#simulator-settings)
- [Actions and rewards](#actions-and-rewards)
- [Fixed PID](#fixed-pid)
- [Learned PID tuning](#learned-pid-tuning)
- [Training](#training)
- [Evaluation and comparison](#evaluation-and-comparison)
- [Testing and GUI behavior](#testing-and-gui-behavior)
- [Batch and staged training](#batch-and-staged-training)
- [Limits and compatibility](#limits-and-compatibility)

## Loading and Overrides

| Top-level section | Purpose |
| --- | --- |
| `common` | Shared environment, simulator, actions, reward, checkpoint, and PID settings. |
| `training` | Overrides used by `stonefish_rl_training.py single` and the base configuration for multi-training. |
| `evaluation` | Overrides used by both evaluation scripts and the Evaluate GUI. |
| `testing` | Overrides used by the docking CLI test and Test GUI. |

Each workflow loads `common`, then recursively merges its own section. Other
workflow sections are excluded. Individual nested keys override shared keys;
lists replace entire lists, not selected elements. Empty lists clear inherited
lists; an empty mapping does not clear an inherited mapping. Do not use `null`
to disable a required section. Unknown keys are not generally validated and may
be silently unused. Use YAML booleans `true`/`false`, not quoted strings.

For example, keep noise enabled in `common.sim` and disable it only for testing:

```yaml
testing:
  sim:
    enable_offset_noise: false
    enable_random_noise: false
```

Most paths resolve relative to the repository root; absolute paths also work.
Run these commands from the repository root:

```bash
python3 src/python/stonefish_rl_gui.py
python3 src/python/stonefish_rl_training.py single include/parameters/stonefish_rl.yaml
python3 src/python/stonefish_rl_evaluation.py include/parameters/stonefish_rl.yaml
python3 src/python/utils/test_docking_env.py include/parameters/stonefish_rl.yaml
```

The three docking CLI scripts above use the unified file when its path is omitted.
`evaluate_docking.py` also accepts it, but use `stonefish_rl_evaluation.py` for
the evaluation-result files and metadata consumed by the plotter.
Old flat YAMLs remain supported when explicitly selected, with their existing
behavior. Generated run configurations are resolved flat YAMLs.

## Environment and Observations

Place shared fields under `common.env`. Override them under a workflow's `env`
only when that difference is intentional.

| Key | Meaning and supported values |
| --- | --- |
| `type` | `docking` for direct commands; `pid_tuning` for learned PID parameters. The environment factory also supports `football`, which needs its own compatible configuration. The legacy docking CLI test directly constructs the docking environment, so use the Test GUI for PID-tuning checkpoints. |
| `state_config` | Path to the raw Stonefish observation specification YAML. Determines raw fields, their names, order, and relative-pose entities. |
| `action_config` | Path to the Stonefish actuator specification YAML. Determines actuator ordering and setpoint limits. |
| `Inverse` | Boolean switch for the Python pose inversion path. `true` enables station-frame pose observations; `false` uses the non-inverting path. Case-sensitive key. |
| `observation_frame` | Legacy alternative: `auv` or `docking_station`. Used only when `Inverse` is absent. |
| `history_length` | Number of additional historical frames. `0` means only the current frame; `2` means three frames total. Nonpositive values are treated as zero. |
| `history_gap` | Seconds between sampled history frames. `0` uses consecutive observations; positive gaps round to simulator-control steps, with a minimum spacing of one step. |
| `observe_actions` | Append the applied command vector to each observation frame. In PID tuning this is the underlying 4/6-axis command, not the 16 learned parameters. |
| `yaw_observation` | `angle`: yaw divided by pi; `cos_sin`: replace one yaw field with cosine and sine, adding one value per frame. |
| `episode_duration` | Episode time limit in seconds. Current overrides: training 120, evaluation 60, testing 600. |
| `rl_observation_freq` | Control/observation frequency in Hz; also determines PID timestep `dt = 1 / frequency`. Must be positive. |
| `instances` | Training worker count; CLI test worker count; default comparison worker cap in unified evaluation configurations. See below. |
| `base_port` | Base simulator communication port. Instance ranks and comparison `port_stride` produce additional ports. Avoid overlap between simultaneous runs. |

### Instance counts

Training starts `training.env.instances` workers plus `training.train.eval_instances`
evaluation workers (one if omitted). That callback uses the resolved **training** configuration, not the
standalone `evaluation` section.

Unified testing and evaluation default to one instance, even if `common.env`
contains a larger count. Set the count in the corresponding workflow explicitly.
The interactive Test GUI and simple single-model evaluator always operate one
environment. Parallel comparison runs different models concurrently, not several
episodes of one model at once. Existing flat YAMLs do not receive these new defaults.

### Pose inversion and ordering

For the explicit PID station-frame path, use:

```yaml
common:
  env:
    type: pid_tuning
    state_config: include/observations/ds_state_pid_pose.yaml
    Inverse: true
```

The first four raw fields must stay `error_x`, `error_y`, `error_z`, `error_yaw`.
The additional `relative_roll` and `relative_pitch` fields are located by name;
keep those names and do not rename them to start with `error`, because reward
code uses name-pattern selection. Reordering fields changes checkpoint semantics
even when the vector length stays unchanged.

The pose is composed as `[x, y, z, roll, pitch, yaw]`, with rotation
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Inversion uses `R.T` and `-R.T @ translation`.
Noise is applied **after** inversion, followed by position normalization and yaw
encoding. Other fields, such as velocities, are not rotated by this pose helper.
Position observations are clipped to +/-5 metres and divided by 5. Roll/pitch
remain in radians. Inversion itself does not add vector entries; adding roll/pitch
to the source specification does.

The supplied default instead uses `ds_state_v2_InDSFrame_config.yaml`, which
already swaps the relative-pose entities, with `Inverse: false`. Do not invert
that source again. This is not identical to the explicit Python inversion path:
`Inverse: false` still selects the legacy offset-processing/PID branch internally.
For station-frame PID use the explicit configuration above.

Observation size is `(raw fields + yaw extra + appended command size) *
(history_length + 1)`. `yaw extra` is 1 for `cos_sin`, otherwise 0. At zero history
gap, history is oldest-to-newest; at positive gap, the current frame is first.
Changing this layout requires a compatible checkpoint or new training.

## Simulator Settings

Set these under `common.sim` or a workflow's `sim` override.

| Key | Meaning |
| --- | --- |
| `scene_path` | Stonefish `.scn` scenario, including the robot and docking station. |
| `resources_path` | Resource directory passed to the simulator. |
| `graphical_interface` | Whether to launch the simulator with graphics. The Test GUI's checkbox overrides this for its session. |
| `evaluation_graphical_interface` | Training callback option: true shows only the first evaluation worker; all other evaluation workers are headless. False makes every evaluation worker headless, independently of training graphics. |
| `resolution` | Numeric resolution argument passed to the simulator launcher. |
| `log_mesh_loading` | Print Stonefish mesh-loading progress. Set `false` in `common.sim` to suppress geometry-path and face-count/timing messages while retaining errors and other logs. Omission preserves logging. Requires `patches/stonefish-mesh-logging.patch` in the Stonefish runtime; the launcher sets `STONEFISH_LOG_MESH=0` or `1` for each new simulator process. |
| `current` | Enable water current; false sends a zero current vector. |
| `current_value` | Two horizontal components `[x, y]`. With random sampling these are nonnegative component bounds; otherwise they are the fixed components. Vertical current is zero. |
| `current_uniform` | True samples each component uniformly from `[-bound, +bound]` at reset; false uses the fixed `current_value`. This controls sampling, not a spatial current-field model. |
| `enable_offset_noise` | Enable persistent/accumulating observation error, initialized per episode. |
| `enable_random_noise` | Enable per-observation random perturbations. |
| `enable_noise` | Legacy fallback for either noise switch when that explicit switch is absent. Explicit switches take precedence. |

Noise distributions, distance/visibility scaling, reset locations, and vertical
current behavior are implemented in the environment, not exposed as extra YAML
knobs. The supplied testing override disables currents and both noise types;
evaluation keeps the shared noisy/current-enabled setup.

## Actions and Rewards

Configure faults in `common.action` (inherited by training/evaluation/testing):

```yaml
use_thruster_health: true
thruster_health:
  min: 0.2
  max: 1.0
  max_affected: 2
```

With health enabled, each reset randomly selects between one and `max_affected`
distinct actuators from the full actuator list, without the old horizontal/vertical
group restriction. Each selected actuator receives an independent uniform health
value between `min` and `max`; all others stay at 1. Equal bounds give a fixed
value. Health is a command multiplier: 0 disables the actuator, 0.5 retains half
its command, and 1 is healthy. Bounds must satisfy `0 <= min <= max <= 1`.
`max_affected` must be an integer from 0 through the number of actuators. Setting
it to 0 keeps all healthy while retaining the observation suffix; setting it to
5 allows all five to be affected. A selected value of 1 is still healthy.

The vector follows actuator-YAML order (five entries for Girona1000), and scales
commands after TCM allocation. Sampling uses the environment's seeded random
generator. Omitted settings default to min 0.2, max 1.0, and at most two actuators.

Health remains fixed throughout the episode. Its vector is appended once after
the complete observation history, even when action observations are disabled.
The size becomes `frame_size * (history_length + 1) + actuator_count`.
Health is not normalized, noised, or included in historical frames. With the switch
false (the default), commands and observation size retain their original behavior.
Enabling it requires a checkpoint trained with this additional observation suffix.

`common.action.force_6Dof: true` enables allocation through `tcm`. Despite the
name, the number of policy commands is the number of TCM rows: the supplied
four rows mean `[x, y, z, yaw]`. A six-row PID TCM uses
`[x, y, z, roll, pitch, yaw]`. Columns must match the actuator specification order.
The mapping is `setpoints = action @ tcm`; the supplied matrix produces five
thruster setpoints. These are normalized commands, not calibrated Newton forces.
With `force_6Dof: false`, the policy commands actuators directly and TCM is unused.
Fixed PID and PID tuning require allocation enabled and a four- or six-row TCM.

In the referenced actuator YAML, `action_config.specs` entries contain
`actuator_name`, `action_type`, `output_name`, `min_value`, and `max_value`.
In the observation YAML, `state_config.specs` entries select `entity_name`,
`field_type`, `component`, and `output_name`; relative-pose entries also specify
`entity_1` and `entity_2`. Match these to the scene and preserve required names.

`common.reward.weights` contains seven weights for the active
`calculate_additional_reward_v1_2` implementation:

| Index | Term |
| --- | --- |
| 0 | Horizontal x/y error |
| 1 | Vertical z error |
| 2 | Yaw error |
| 3 | Docking alignment/accuracy |
| 4 | Change in command, for smoothing |
| 5 | Velocity penalty |
| 6 | Collision penalty |

Use zero to disable a weighted term; replace the complete seven-value list when
overriding it. Success bonuses, goal tolerances, visibility conditions, and reward
formula selection are not controlled by these weights. Alternate reward functions
exist in code but are not selectable by a YAML key.

## Fixed PID

Configure `common.pid`; gains and limits are four-value arrays ordered
`[x, y, z, yaw]`. Distances are metres, angles radians, and time seconds.

| Key | Meaning |
| --- | --- |
| `mode` | `one_stage` enables all axes immediately. `two_stage` starts with x/y/yaw and delays z. |
| `alignment_threshold` | Positive horizontal distance threshold in metres. Once measured `sqrt(x*x + y*y)` is below it, z is enabled for the rest of the episode. |
| `z_offset` | Added to the z control error; default 0.5 m biases the commanded docking depth. Does not move the environment's success goal. |
| `target_pose` | Desired AUV pose in station coordinates: `[x, y, z, roll, pitch, yaw]`. The station-frame controller uses target x/y/z/yaw; roll/pitch targets are not controlled. Unused by the legacy AUV-frame branch. |
| `kp` | Nonnegative proportional gains. |
| `ki` | Nonnegative integral gains. |
| `kd` | Nonnegative derivative gains. |
| `output_limit` | Nonnegative per-axis command magnitude limits. Final actions are also clipped to the environment action bounds. |
| `integral_limit` | Nonnegative bounds on the accumulated error integral, before multiplication by `ki`. |

The controller computes `u = kp*error + ki*integral + kd*derivative`. The integral
accumulates `error*dt`, is bounded, and rejects updates that worsen saturation.
Derivatives use measured errors before target offsets, avoiding derivative spikes
when learned offsets change. Yaw differences wrap across +/-pi.

The stage switch uses measured horizontal error before offsets and remains latched
until reset. x/y/yaw continue correcting after z activates. Inactive z has zero
command and no accumulating integral; it does not actively hold depth. Derivative
output is suppressed on the first sample and when an axis first becomes active.
Controller history resets each episode.

In the station-frame path, error is target minus observed pose. Linear commands
and yaw torque are rotated back into the AUV frame before allocation. A four-axis
TCM cannot apply the roll/pitch torque components of that rotated yaw command.

For a fixed PID baseline, set `evaluation.evaluate.algorithm: PID` and
`model_path: PID`, or select **fixed PID** in the Test GUI. No checkpoint is needed.
Either the algorithm or model-path value `PID` selects the fixed baseline.

## Learned PID Tuning

Set `common.env.type: pid_tuning`. The policy still observes the docking state,
but produces 16 normalized actions in this order:

```text
kp_x kp_y kp_z kp_yaw
ki_x ki_y ki_z ki_yaw
kd_x kd_y kd_z kd_yaw
offset_x offset_y offset_z offset_yaw
```

`common.pid_tuning` has four blocks: `kp`, `ki`, `kd`, `offset`. Every block needs
four-value `low` and `high` arrays, ordered x/y/z/yaw. Values must be finite with
`low <= high`; gains must be nonnegative, while offsets may be negative. Equal
bounds fix that parameter. Position offsets are metres and yaw offset is radians.

Every step maps clipped policy output `a` from `[-1, 1]` into these bounds:
`parameter = low + (a + 1)/2 * (high - low)`. Zero action means the midpoint,
not necessarily zero gain. Learned gains/offsets replace the fixed PID values on
each step, including `pid.z_offset`; the stage mode and PID limits still apply.
The persistent controller is not reset at every policy step.

Train with PPO/SAC/TD3 as usual. To test the resulting checkpoint, keep
`env.type: pid_tuning`, choose its actual algorithm (for example PPO), and supply
the `.zip`/`.onnx` path. **Do not label a PID-tuning checkpoint as `PID`**, which
selects the fixed controller instead. The GUI exposes `pid_parameters` and
`pid_command` through its info/plot channels.

## Training

### Model settings

These live under `training.model`.

| Key | Meaning and consumer |
| --- | --- |
| `algorithm` | `PPO`, `SAC`, or `TD3`; use these uppercase spellings. Separate from the evaluation checkpoint algorithm. |
| `policy` | Policy class name passed to a newly constructed model, normally `MlpPolicy`. |
| `pretrained` | True loads an existing checkpoint; false constructs a new model. |
| `model_path` | Checkpoint to load when `pretrained` is true. Separate from `common.evaluate.model_path`. |
| `weights_only` | Forwarded to the model's `load` call. The launcher does not implement a separate weights-only transfer/reinitialization procedure; do not assume this resets optimizer or checkpoint settings. |
| `learning_rate_start`, `learning_rate_end` | Linear learning-rate schedule for new models; equal values give a constant rate. Values may be numeric or `auto`. |
| `learning_rate` | Optional constant rate, taking precedence over start/end. `auto` resolves to 0.0003 for PPO/SAC and 0.001 for TD3. |
| `n_steps` | PPO rollout steps per environment. Total rollout samples are `n_steps * training.env.instances`. |
| `batch_size` | Passed to new PPO and TD3 models. Currently not passed by the SAC branch, which uses its library default. For PPO use a positive value compatible with the rollout size, preferably a divisor. |
| `clip_range` | PPO clipping parameter for new models. |
| `buffer_size` | Replay-buffer capacity for new SAC and TD3 models. |
| `gradient_steps` | Passed to new SAC models only. |
| `learning_starts` | SAC steps before updates begin; not currently forwarded by the TD3 branch. |
| `ent_coef` | SAC entropy coefficient, numeric or `auto`. Not currently forwarded to PPO. |
| `policy_delay` | TD3 policy update delay. |

The template starts fresh training with PPO. Resume branches load checkpoint
hyperparameters rather than forwarding the new-model YAML arguments above.
Changing those fields is therefore not a general way to retune a resumed model.

### Training duration and outputs

| `training.train` key | Meaning |
| --- | --- |
| `total_timesteps` | Requested SB3 training transitions, aggregated across workers. Algorithms may complete a rollout beyond this target. |
| `seed` | Starting integer seed for standalone training (default configuration: 42). Controls Python, NumPy, PyTorch/SB3, environment resets and action spaces. `null` disables explicit seeding. Multi-training assigns concrete seeds from `starting_seed`. |
| `reset_num_timesteps` | Whether `learn()` resets its timestep counter, particularly relevant when resuming. |
| `eval_freq` | Evaluation-callback frequency in vectorized step calls; not automatically divided by worker count. Roughly `eval_freq * instances` training transitions between callbacks. |
| `eval_instances` | Positive number of parallel evaluation worker processes/simulators; defaults to 1 when omitted. Separate from training workers. |
| `eval_episodes` | Positive total number of episodes evaluated per callback, distributed across evaluation workers; defaults to 5 when omitted. |
| `success_rate_threshold` | Count complete evaluations strictly above this fraction. `0.90` requires at least 23/25 successes. `null` or omission disables both success-based stop rules. |
| `success_rate_required_evaluations` | Number of qualifying evaluations to end a stage; default 3, cumulative rather than consecutive. The counter resets per stage. |
| `success_rate_drop_threshold` | After the first qualifying evaluation, a rate strictly below this floor ends the stage; default 0.60 (capped at the high threshold). Save and advance normally. |
| `save_path` | Directory for the final model. |
| `save_name` | Name component in the generated timestamped model filename. |
| `warm_start_path` | Optional filename prefix; despite its name, not the input checkpoint path. Defaults to the algorithm prefix. |
| `output_model_path` | Optional explicit final output path, used by multi-stage training. Overrides generated naming. |
| `stop_file` | Optional file watched to request graceful training termination. |
| `stop_check_freq` | Callback-call interval for checking `stop_file`; default 100. |

Best evaluation-callback checkpoints are written to `log.log_dir/best_model.zip`,
not necessarily to `train.save_path`. The template's evaluation path points there.
Set `eval_instances: 5` and `eval_episodes: 25` for five episodes per worker.
With `success_rate_threshold: 0.90`, a stage ends after three complete evaluations
with at least 23 successes out of 25 (not necessarily consecutive). After the first
such result, fewer than 15 successes out of 25 also ends the stage. Exactly 60%
does not trigger the drop rule. Initial low success does not stop learning.
The step budget remains an independent stopping condition. Docking success requires completing the existing docking hold;
a timeout or briefly entering the goal region is not a success. This check runs
after every evaluation, independently of reward improvement. The current model
is saved through the normal final-save path, and multistage training loads that
checkpoint for the next stage. The final stage simply finishes the run.
Stages may override this setting with `train: {success_rate_threshold: null}`
to use only their step budget. This does not use the GUI stop-request file,
which has different skip-stage behavior.
For other totals, SB3 distributes the counted episodes as evenly as possible;
use at least as many episodes as workers to avoid workers with no assigned episodes.
Workers run in separate processes, not Python threads. Training waits for the
evaluation callback to finish. Evaluation ports start at `base_port + training
instances + 1` and increase by one per worker. Reserve these ports and account
for the extra simulator CPU/memory cost. Set both graphical flags false for
headless training and evaluation. Evaluation monitor output uses a separate
`evaluation.monitor.csv` file in the log directory.

`evaluation.evaluate.num_episodes` does not configure the training callback.

Training worker `i` uses `train.seed + i`. Evaluation worker `i` uses
`train.seed + training_instances + 1 + i`, keeping its stream separate from
training. Only the first reset is seeded; later episodes advance their RNGs, so
they do not repeat the same initial conditions. Evaluation RNGs also advance
between callbacks. Python domain randomization uses those same seeded environment
RNGs. New models and loaded checkpoints both receive the configured model seed.

Set `starting_seed` in `multi_train.yaml` for a multi-training job (or omit it to
inherit `training.train.seed`). Each stage/run receives the next block of
`training_instances + 1 + eval_instances` seeds, including stages skipped by early
stopping. With 20 training workers, five evaluation workers and starting seed 42,
the five stage seeds are **42, 68, 94, 120, 146**; another run starts at 172.
The generated stage YAMLs and console output record the exact seeds. This is a
reproducible sequence of stage streams, not restoration of a checkpoint's RNG
state. Changing worker counts or stage structure changes the allocation.

The seed does not control Stonefish's separate C++ sensor-noise generator.
Physics/GPU execution may also differ between runs; identical seeds are not a
guarantee of bit-for-bit training results.

| `training.log` key | Meaning |
| --- | --- |
| `enable_wandb` | Enable Weights & Biases integration; currently true, inherited by all multi-training stages. |
| `project_name` | W&B project. |
| `run_name` | W&B run identifier; `default` generates an algorithm/policy/timestamp identifier. |
| `tags` | List of W&B tags. |
| `log_dir` | Training/evaluation logs and best-model directory; also contains W&B/TensorBoard subdirectories when used. |

Each training invocation records its resolved configuration in W&B, including its
seed and domain-randomization settings. With `run_name: default`, stages create
separate runs using timestamps with microsecond precision. Authenticate with
`wandb login` before launching training if this machine is not already logged in.

## Evaluation and Comparison

`common.evaluate` selects the checkpoint shared by evaluation and the Test GUI's
trained-model mode. `evaluation.evaluate` adds result/run settings or overrides
that selection.

| `evaluate` key | Meaning |
| --- | --- |
| `algorithm` | `PPO`, `SAC`, `TD3`, `ONNX`, or `PID`. |
| `model_path` | Compatible checkpoint path, or `PID` for a fixed baseline. |
| `num_episodes` | Number of evaluation episodes per model. |
| `step_per_print` | Progress-print interval. The comparison evaluator supports 0 to suppress periodic output; use a positive value with the legacy simple evaluator. |
| `seed` | Evaluation seed, default 0. Comparison can override it with `compare.seed`. |
| `evaluation_name` | Human-readable evaluation name; blank lets the result writer derive a name. |
| `note` | Optional text stored with the result, not used as a plot label. |
| `evaluation_tag` | Optional evaluation variant tag. |

Names, notes, and tags are result metadata for the comparison evaluator (including
its single-model mode). Each evaluation has a unique `.evaluation.json` record
with summary, episodes, and configuration, so repeating a model/name does not
replace its previous result. Result directory selection is a CLI/GUI control,
not a key in this unified template. The plotter reads those files, not this YAML.

### Comparison options

Set these under `evaluation.compare`.

| Key | Meaning |
| --- | --- |
| `compare` | False uses `evaluate.algorithm/model_path`; true uses the model lists. The Evaluate GUI builds comparison runs from its selected model rows. |
| `algorithms`, `model_paths` | Equal-length lists, paired by index. Paths may include `PID` for a baseline. |
| `parallel` | False runs models sequentially; true allows a worker pool when comparison is enabled. |
| `max_workers` | Optional explicit worker cap; otherwise unified YAML uses `evaluation.env.instances` (default 1). Limited by the number of models in a batch. |
| `batch_size` | Number of models in each scheduling batch; unrelated to SB3 minibatches or episode count. |
| `port_stride` | Increment between allocated model ports. Use positive spacing and avoid ports used by other simulator sessions. |
| `force_headless_parallel` | Forces graphics off in the per-model configuration. Currently applied even if the run is sequential; keep false when you want graphics. |
| `auto_correct_observation` | Attempt to infer compatible history length from checkpoint input size. Does not prove field ordering, frames, or semantics match. False in this template. |
| `num_episodes`, `step_per_print` | Optional comparison-wide overrides of the corresponding evaluation settings. |
| `seed` | Optional seed override, also read by the comparison launcher in single-model mode. |
| `yaw_observation`, `history_length`, `history_gap` | Optional per-model observation overrides, falling back to `env` values. |
| `evaluation_name`, `note`, `evaluation_tag` | Optional per-model metadata. Set these here in comparison mode, rather than relying on single-model `evaluate` metadata. |
| `evaluation_tags` | Legacy alias for `evaluation_tag`, used only if the singular key is absent. |
| `num_models` | Optional legacy count; actual model-list length wins if they differ. |

Per-model observation overrides and metadata accept a scalar for all models, a
one-element list to broadcast, or a list matching the model count. Algorithms and
paths must explicitly have the same length. Counts and scheduling sizes should
be positive integers; history gaps may be zero.

Example override for two concurrent checkpoints:

```yaml
evaluation:
  env:
    instances: 2
  compare:
    compare: true
    parallel: true
    force_headless_parallel: true
    algorithms: [PPO, PPO]
    model_paths: [models/first.zip, models/second.zip]
    evaluation_name: [First model, Second model]
    note: [Baseline training, Training with currents]
```

## Testing and GUI Behavior

The outer `testing` section selects the workflow. Its inner `testing` mapping
contains CLI loop options; the repeated name is intentional for compatibility.

| Key under outer `testing` | Meaning |
| --- | --- |
| `testing.episodes` | CLI loop budget: episodes * episode_duration * frequency steps. It is not a strict count of completed episodes if some terminate early. |
| `testing.step_per_print` | Positive CLI print interval in steps. |
| `controller.logitech` | CLI joystick control; takes priority over random mode. |
| `controller.random` | CLI random actions when joystick mode is off. |
| `controller.force_vector` | Constant command vector when both modes are off; length must match the docking action space. |

The Test GUI selects its action mode interactively: zero, random, manual,
controller, fixed PID, or trained model. Its Start/Stop lifecycle is not governed
by the CLI episode budget. The GUI reloads environment code on Start, and its
graphical checkbox overrides YAML for that session. Use Load to refresh model
defaults after changing the selected YAML; restart the environment to apply changes.

The Evaluate GUI's **Workers** control sets `compare.max_workers`; **Parallel** alone
does not increase that limit. Select at least two workers to evaluate multiple
models concurrently. The run log reports the effective worker count, capped by
the number of selected models and batch size. Use headless mode to avoid several
simulator windows when evaluating concurrently.

The Evaluate GUI Load button reloads configuration without discarding selected
model rows. **Load From YAML** explicitly replaces the model list. Its editor
shows the resolved evaluation configuration, not all workflow sections; edit the
source unified file to make changes shared by training and testing too. GUI run
controls and model rows can override the resolved YAML when generating a run.

## Batch and Staged Training

Batch scheduling remains in `multi_train.yaml` or `multi_train_v2.yaml`; it is not
an extra top-level section of the unified file. Select the unified training base
with `base_config: stonefish_rl.yaml` in a batch YAML in this directory. Base-file
paths in batch YAML are resolved relative to that batch file. `--base-config`
overrides this selection and also resolves relative paths against the batch file's
directory. Shared settings are resolved first, then batch/stage
overrides operate on the resulting flat configuration.

| Batch key | Destination or behavior |
| --- | --- |
| `base_config` | Base training YAML, unified or legacy flat. |
| `training_config_num` | Positive number of training runs. |
| `model_type` | `model.algorithm`. |
| `pre_trained` | Boolean `model.pretrained`. |
| `pretrained` | Checkpoint `model.model_path`, not the boolean switch. |
| `max_iteration` | `train.total_timesteps`. |
| `eval_freq` | `train.eval_freq`; positive integer, one-item list, or one value per run. Counts vectorized step calls. |
| `output_agent_name` | `train.save_name`. |
| `warm_start_path` | Output filename prefix. |
| `reward_weights` | Complete seven-value `reward.weights` vector. |
| `yaw_observation`, `history_length`, `history_gap`, `rl_observation_freq` | Corresponding `env` settings. |
| `multistage.use` | Enable sequential stages for each run. |
| `multistage.stages` | Ordered stage definitions, each with a name and configuration overrides. Later stages load the preceding stage's checkpoint. |

Run settings generally accept one scalar, one-element lists, or one value per
run. Reward weights accept a single complete vector or per-run vectors. Stage
overrides support nested mappings (for example `sim: {current: false}`), path
triples such as `[sim, current, false]`, and comma-separated equivalents.
Stage `max_iteration` sets that stage's timestep budget. Keep checkpoint layouts
compatible between stages. Intermediate model names are stable for chaining;
final models use generated run naming.

Preview without starting simulators:

```bash
python3 src/python/stonefish_rl_training.py multi include/parameters/multi_train_v2.yaml --base-config stonefish_rl.yaml --dry-run --output-dir /tmp/stonefish-config-preview
```

This still writes generated YAMLs to the specified output directory. The batch
control GUI is separate from the Test/Evaluate/Plot tabs.

## Limits and Compatibility

- Keep training and checkpoint evaluation observation fields, order, history,
  encoding, and action layout compatible. Equal vector sizes alone are not enough.
- `Inverse` changes policy observations, not the raw reward/success definitions;
  changing the observation source's entities may also affect those raw calculations.
- PID target/offset edits do not redefine docking success. Goal tolerances, hold
  duration, reset distributions, noise magnitudes, and reward-function selection
  still require environment-code edits. PID yaw remains active in both stages.
- The fixed-PID baseline and the learned PID-tuning policy are different modes;
  choose the checkpoint algorithm for the latter.
- Separate port ranges do not guarantee independent training and test sessions:
  the training launcher currently kills existing `StonefishRL` processes at startup.
- YAML loading is not a full schema validation. Check shapes, positive counts,
  checkpoint existence, scene/entity names, and the effective resolved configuration
  before starting a long run.

Implementation references: [configuration loader](../../src/python/library/configuration.py),
[docking environment](../../src/python/library/docking_env.py),
[PID controller](../../src/python/library/pid_docking.py),
[PID tuning wrapper](../../src/python/library/pid_tuning_env.py),
[trainer](../../src/python/library/training.py), and
[comparison evaluator](../../src/python/library/evaluation.py).

Training and instance benchmarks check a contiguous simulator port range before
launching each stage/trial. If a port is busy, they try the next base port until
the whole range is free, including training evaluation workers. The selected
range is printed; benchmark trial YAMLs record the selected base. The source
YAML is unchanged. This check does not reserve ports against unrelated processes
starting concurrently, and existing Stonefish simulators still block a new run.
