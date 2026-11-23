import argparse
import copy
import datetime as dt
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import yaml
from library.configuration import resolve_config, DEFAULT_CONFIG
from library.training_seeding import allocate_stage_seeds, validate_seed
from library.task_registry import is_reward_vector, validate_task_reward


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MULTI_CONFIG = REPO_ROOT / "include/parameters/multi_train.yaml"
DEFAULT_BASE_CONFIG = REPO_ROOT / DEFAULT_CONFIG
DEFAULT_OUTPUT_DIR = REPO_ROOT / "include/parameters/generated_multi_train"
ACTIVE_STATUS_NAME = "active_training.yaml"


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stop requested\n")


def remove_if_exists(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_active_status(control_dir, data):
    save_yaml(control_dir / ACTIVE_STATUS_NAME, data)


def get_first(config, keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def warn_if_template_is_inactive(multi_config):
    """Warn when a stage template exists but the active multistage key is off."""
    multistage = multi_config.get("multistage", {}) or {}
    template = multi_config.get("multistage_template", {}) or {}
    if (
        isinstance(template, dict)
        and template.get("use", template.get("enabled", False))
        and not bool(multistage.get("use", multistage.get("enabled", False)))
    ):
        print(
            "[WARNING] multistage_template.use is true, but the multi-training command only runs "
            "stages from the active 'multistage' key. Rename 'multistage_template' "
            "to 'multistage' if you want these stages to run."
        )


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_reward_weights(config, context):
    validate_task_reward(config, context)


def value_for_run(value, index, run_count, key, allow_reward_vector=False):
    if allow_reward_vector and is_reward_vector(value):
        return value

    if not isinstance(value, list):
        return value

    if len(value) == run_count:
        return value[index]

    if len(value) == 1:
        return value[0]

    raise ValueError(
        f"'{key}' must be a scalar, a one-item list, or a list with {run_count} items."
    )


def set_nested(config, section, key, value):
    config.setdefault(section, {})
    config[section][key] = value


def parse_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def set_by_path(config, path, value):
    if isinstance(path, str):
        parts = [item for item in path.replace(",", ".").split(".") if item]
    else:
        parts = [str(item) for item in path]
    if not parts:
        raise ValueError("Stage override path cannot be empty.")

    if parts[0] in ("max_iteration", "max_iterations", "max_iter", "max_iters"):
        parts = ["train", "total_timesteps"]

    if parts == ["sim", "enable_noise"]:
        parsed_value = parse_scalar(value)
        config.setdefault("sim", {})
        config["sim"]["enable_offset_noise"] = parsed_value
        config["sim"]["enable_random_noise"] = parsed_value
        config["sim"].pop("enable_noise", None)
        return

    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = parse_scalar(value)


def apply_stage_override(config, override):
    """Apply one stage override.

    Supported forms:
      - "sim,current,false"
      - ["sim", "current", false]
      - ["sim", "enable_noise", true]  # legacy alias for both split noise flags
      - {"max_iteration": 100000}
      - {"sim.current": false}
      - {"sim": {"current": false, "enable_offset_noise": true, "enable_random_noise": true}}
    """
    if isinstance(override, str):
        parts = [part.strip() for part in override.split(",")]
        if len(parts) < 3:
            raise ValueError(f"Invalid stage override string: {override!r}")
        set_by_path(config, parts[:-1], parts[-1])
        return

    if isinstance(override, (list, tuple)):
        if len(override) < 3:
            raise ValueError(f"Invalid stage override list: {override!r}")
        set_by_path(config, override[:-1], override[-1])
        return

    if isinstance(override, dict):
        for key, value in override.items():
            if key in ("name", "stage_name"):
                continue
            if key in ("overrides", "set", "params", "parameters"):
                apply_stage_overrides(config, value)
            elif isinstance(value, dict) and "." not in str(key) and "," not in str(key):
                for child_key, child_value in value.items():
                    set_by_path(config, [key, child_key], child_value)
            else:
                set_by_path(config, key, value)
        return

    raise ValueError(f"Unsupported stage override: {override!r}")


def apply_stage_overrides(config, overrides):
    if overrides is None:
        return
    if isinstance(overrides, (str, dict)):
        apply_stage_override(config, overrides)
        return
    for override in overrides:
        apply_stage_override(config, override)


def multistage_enabled(multi_config):
    multistage = multi_config.get("multistage", {}) or {}
    return bool(multistage.get("use", multistage.get("enabled", False)))


def normalize_stage_definition(stage_item, index):
    if isinstance(stage_item, dict):
        if "name" in stage_item:
            name = str(stage_item.get("name") or f"stage_{index + 1:02d}")
            overrides = []
            for key, value in stage_item.items():
                if key == "name":
                    continue
                if key in ("overrides", "set", "params", "parameters"):
                    if isinstance(value, list):
                        overrides.extend(value)
                    else:
                        overrides.append(value)
                else:
                    overrides.append({key: value})
            return name, overrides
        if len(stage_item) == 1:
            name, overrides = next(iter(stage_item.items()))
            return str(name), overrides
    return f"stage_{index + 1:02d}", stage_item


def normalize_multistage(multi_config):
    multistage = multi_config.get("multistage", {}) or {}
    if not multistage_enabled(multi_config):
        return []

    raw_stages = multistage.get("stages")
    if raw_stages is None:
        raw_stages = [
            {key: value}
            for key, value in multistage.items()
            if str(key).lower().startswith("stage")
        ]

    if not raw_stages:
        raise ValueError("multistage.use is true, but no stages were defined.")

    stages = []
    for index, stage_item in enumerate(raw_stages):
        name, overrides = normalize_stage_definition(stage_item, index)
        stages.append({
            "name": name,
            "overrides": overrides,
        })
    return stages


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "model"


def with_zip_suffix(path):
    path = Path(path)
    if path.suffix != ".zip":
        return path.with_suffix(".zip")
    return path


def model_output_basename(config):
    train_cfg = config.get("train", {}) or {}
    model_cfg = config.get("model", {}) or {}
    prefix = train_cfg.get("warm_start_path", model_cfg.get("algorithm", "") + "_")
    save_name = train_cfg.get("save_name", "model")
    return safe_name(f"{prefix}{save_name}")


def build_stage_configs(
    run_config,
    stages,
    run_number,
    generated_dir,
):
    """Build sequential stage configs for one run."""
    base_name = model_output_basename(run_config)
    intermediate_dir = REPO_ROOT / "models" / "multistage_intermediate" / generated_dir.name / f"run_{run_number:03d}_{base_name}"
    stage_configs = []
    previous_output = None

    for stage_index, stage in enumerate(stages):
        stage_number = stage_index + 1
        stage_config = copy.deepcopy(run_config)
        apply_stage_overrides(stage_config, stage.get("overrides", []))
        validate_reward_weights(stage_config, f"stage {stage_number} ({stage['name']})")

        if previous_output is not None:
            stage_config.setdefault("model", {})
            stage_config["model"]["pretrained"] = True
            stage_config["model"]["model_path"] = str(previous_output)

        is_final_stage = stage_number == len(stages)
        if is_final_stage:
            output_model_path = None
        else:
            output_model_path = with_zip_suffix(intermediate_dir / f"{base_name}_{stage['name']}")

        stage_config.setdefault("train", {})
        if output_model_path is None:
            stage_config["train"].pop("output_model_path", None)
        else:
            stage_config["train"]["output_model_path"] = str(output_model_path)
        stage_config["train"].setdefault("reset_num_timesteps", False)

        stage_configs.append({
            "stage_number": stage_number,
            "stage_count": len(stages),
            "stage_name": stage["name"],
            "config": stage_config,
            "output_model_path": output_model_path,
        })
        previous_output = output_model_path

    return stage_configs


def build_run_config(base_config, multi_config, run_index, run_count):
    config = copy.deepcopy(base_config)
    # Task-specific settings need no new launcher keys. Common overrides first,
    # then one optional mapping per run, followed by the established sweep fields.
    apply_stage_overrides(config, multi_config.get('overrides'))
    run_overrides = multi_config.get('run_overrides')
    if run_overrides is not None:
        if not isinstance(run_overrides, list) or len(run_overrides) != run_count:
            raise ValueError('run_overrides must contain one mapping per training run.')
        if not all(isinstance(item, dict) for item in run_overrides):
            raise ValueError('Every run_overrides entry must be a mapping.')
        apply_stage_overrides(config, run_overrides[run_index])
    current_pre_trained = None

    reward_weights = get_first(multi_config, ["reward_weights", "reward_weights_list"])
    max_iteration = get_first(
        multi_config,
        ["max_iteration", "max_iterations", "max_iter", "max_iters"],
    )
    model_type = get_first(multi_config, ["model_type", "model_types"])
    pre_trained = get_first(multi_config, ["pre_trained", "pretrained_enabled"])
    pretrained = get_first(multi_config, ["pretrained", "pretrained_model", "model_path"])
    output_agent_name = get_first(
        multi_config,
        ["output_agent_name", "output_agent_names", "save_name", "save_names"],
    )
    warm_start_path = get_first(
        multi_config,
        ["warm_start_path", "warm_start_paths", "name_prefix", "name_prefixes"],
    )
    yaw_observation = get_first(
        multi_config,
        ["yaw_observation", "yaw_observations", "yaw_observation_mode"],
    )
    history_length = get_first(
        multi_config,
        ["history_length", "history_lengths", "observation_history", "observation_history_length"],
    )
    history_gap = get_first(
        multi_config,
        ["history_gap", "history_gaps", "observation_history_gap"],
    )
    rl_observation_freq = get_first(
        multi_config,
        ["rl_observation_freq", "rl_observation_freqs", "observation_frequency"],
    )

    if reward_weights is not None:
        weights = value_for_run(
            reward_weights,
            run_index,
            run_count,
            "reward_weights",
            allow_reward_vector=True,
        )
        if not is_reward_vector(weights):
            raise ValueError(
                "'reward_weights' entries must be nonempty vectors of finite numbers."
            )
        set_nested(config, "reward", "weights", weights)
        validate_reward_weights(config, 'Training run')

    if max_iteration is not None:
        config.setdefault("train", {})
        config["train"]["total_timesteps"] = value_for_run(
            max_iteration,
            run_index,
            run_count,
            "max_iteration",
        )

    if "eval_freq" in multi_config:
        eval_freq = value_for_run(multi_config["eval_freq"], run_index, run_count, "eval_freq")
        if isinstance(eval_freq, bool) or not isinstance(eval_freq, int) or eval_freq < 1:
            raise ValueError("'eval_freq' must contain positive integers.")
        set_nested(config, "train", "eval_freq", eval_freq)

    if model_type is not None:
        config.setdefault("model", {})
        config["model"]["algorithm"] = value_for_run(
            model_type,
            run_index,
            run_count,
            "model_type",
        )

    if pre_trained is not None:
        config.setdefault("model", {})
        current_pre_trained = value_for_run(
            pre_trained,
            run_index,
            run_count,
            "pre_trained",
        )
        config["model"]["pretrained"] = current_pre_trained

    if pretrained is not None and current_pre_trained is not False:
        config.setdefault("model", {})
        config["model"]["model_path"] = value_for_run(
            pretrained,
            run_index,
            run_count,
            "pretrained",
        )

    if output_agent_name is not None:
        config.setdefault("train", {})
        config["train"]["save_name"] = value_for_run(
            output_agent_name,
            run_index,
            run_count,
            "output_agent_name",
        )

    if warm_start_path is not None:
        config.setdefault("train", {})
        config["train"]["warm_start_path"] = value_for_run(
            warm_start_path,
            run_index,
            run_count,
            "warm_start_path",
        )

    if yaw_observation is not None:
        config.setdefault("env", {})
        config["env"]["yaw_observation"] = value_for_run(
            yaw_observation,
            run_index,
            run_count,
            "yaw_observation",
        )

    if history_length is not None:
        config.setdefault("env", {})
        config["env"]["history_length"] = int(value_for_run(
            history_length,
            run_index,
            run_count,
            "history_length",
        ))

    if history_gap is not None:
        config.setdefault("env", {})
        config["env"]["history_gap"] = float(value_for_run(
            history_gap,
            run_index,
            run_count,
            "history_gap",
        ))

    if rl_observation_freq is not None:
        config.setdefault("env", {})
        config["env"]["rl_observation_freq"] = int(value_for_run(
            rl_observation_freq,
            run_index,
            run_count,
            "rl_observation_freq",
        ))

    validate_reward_weights(config, 'Training run')
    return config


def send_signal_to_process_tree(process, signal_value):
    """Send a signal to the training subprocess group."""
    try:
        os.killpg(os.getpgid(process.pid), signal_value)
    except Exception:
        process.send_signal(signal_value)


def run_training(config_path, stop_timeout, stop_file=None):
    command = [
        sys.executable,
        "-m",
        "stonefish_rl_training",
        "single",
        str(config_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT / "src/python",
        start_new_session=True,
    )

    try:
        return process.wait(), False
    except KeyboardInterrupt:
        print(
            "\n[INFO] Ctrl+C received. Stopping the current training and "
            "continuing with the next one..."
        )
        print("[INFO] Press Ctrl+C again now to abort all remaining trainings.")
        if stop_file is not None:
            print(f"[INFO] Requesting graceful stop through: {stop_file}")
            touch(stop_file)
        else:
            send_signal_to_process_tree(process, signal.SIGINT)

        try:
            return process.wait(timeout=stop_timeout), True
        except subprocess.TimeoutExpired:
            print(
                f"[WARNING] Training did not stop within {stop_timeout} seconds. "
                "Terminating it."
            )
            send_signal_to_process_tree(process, signal.SIGTERM)
            try:
                return process.wait(timeout=15), True
            except subprocess.TimeoutExpired:
                print("[WARNING] Training still running. Force killing it.")
                send_signal_to_process_tree(process, signal.SIGKILL)
                return process.wait(), True
        except KeyboardInterrupt:
            print("\n[INFO] Second Ctrl+C received. Aborting multi-training.")
            send_signal_to_process_tree(process, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                send_signal_to_process_tree(process, signal.SIGKILL)
                process.wait()
            raise


def launch_control_gui(control_dir):
    """Launch the small control GUI for this multi-training session."""
    command = [
        sys.executable,
        "-m",
        "stonefish_rl_gui",
        "--multi-control",
        str(control_dir),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT / "src/python",
            start_new_session=True,
        )
        print(f"[INFO] Multi-train control GUI launched for: {control_dir}")
        return process
    except Exception as exc:
        print(f"[WARNING] Could not launch multi-train control GUI: {exc}")
        return None


def run_multi_training(args):
    multi_config_path = Path(args.multi_config).expanduser().resolve()
    multi_config = load_yaml(multi_config_path)

    run_count = get_first(
        multi_config,
        ["training_config_num", "training_confi_num", "training_num", "num_trainings"],
    )
    if not isinstance(run_count, int) or run_count <= 0:
        raise ValueError("'training_config_num' must be a positive integer.")

    base_config_path = args.base_config or multi_config.get("base_config")
    if base_config_path is None:
        base_config_path = DEFAULT_BASE_CONFIG
    else:
        base_config_path = Path(base_config_path).expanduser()
        if not base_config_path.is_absolute():
            base_config_path = (multi_config_path.parent / base_config_path).resolve()

    base_config = resolve_config(load_yaml(base_config_path), "training")
    next_seed = validate_seed(multi_config.get("starting_seed", base_config.get("train", {}).get("seed")))
    warn_if_template_is_inactive(multi_config)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() / timestamp
    print(f"[INFO] Base config: {base_config_path}")
    print(f"[INFO] Multi config: {multi_config_path}")
    print(f"[INFO] Generated configs: {output_dir}")
    control_dir = output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Control directory: {control_dir}")
    print("[INFO] Multi-train GUI can use this directory to stop the active run.")
    write_active_status(control_dir, {
        "status": "starting",
        "run_count": run_count,
        "base_config": str(base_config_path),
        "multi_config": str(multi_config_path),
        "generated_config_dir": str(output_dir),
        "control_dir": str(control_dir),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    })
    gui_process = None
    if not args.dry_run and not args.no_gui:
        gui_process = launch_control_gui(control_dir)

    stages = normalize_multistage(multi_config)
    if stages:
        print(f"[INFO] Multistage training enabled with {len(stages)} stages.")

    prepared_run_configs = []
    for index in range(run_count):
        try:
            planned_config = build_run_config(base_config, multi_config, index, run_count)
            prepared_run_configs.append(planned_config)
        except Exception as exc:
            prepared_run_configs.append(exc)

    for index in range(run_count):
        run_number = index + 1
        write_active_status(control_dir, {
            "status": "preparing_run",
            "run_number": run_number,
            "run_count": run_count,
            "base_config": str(base_config_path),
            "multi_config": str(multi_config_path),
            "generated_config_dir": str(output_dir),
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        })
        prepared_config = prepared_run_configs[index]
        if isinstance(prepared_config, Exception):
            write_active_status(control_dir, {
                "status": "config_error",
                "run_number": run_number,
                "run_count": run_count,
                "base_config": str(base_config_path),
                "multi_config": str(multi_config_path),
                "generated_config_dir": str(output_dir),
                "error": str(prepared_config),
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            })
            print(f"[ERROR] Could not build training {run_number}/{run_count}: {prepared_config}")
            return 1
        run_config = copy.deepcopy(prepared_config)
        if stages:
            training_items = build_stage_configs(
                run_config,
                stages,
                run_number,
                output_dir,
            )
        else:
            training_items = [{
                "stage_number": None,
                "stage_count": None,
                "stage_name": None,
                "config": run_config,
                "output_model_path": None,
            }]

        # Allocate all stages up front, so early success/user skips do not change
        # the seeds assigned to subsequent runs. Stage YAMLs store concrete seeds.
        next_seed = allocate_stage_seeds([item['config'] for item in training_items], next_seed)
        skip_remaining_stages = False
        for item in training_items:
            stage_number = item["stage_number"]
            stage_count = item["stage_count"]
            stage_name = item["stage_name"]
            current_config = item["config"]
            stage_suffix = f"_stage_{stage_number:02d}" if stage_number is not None else ""
            stop_file = control_dir / f"stop_run_{run_number:03d}{stage_suffix}.request"
            remove_if_exists(stop_file)
            current_config.setdefault("train", {})
            current_config["train"]["stop_file"] = str(stop_file)
            current_config["train"].setdefault("stop_check_freq", 100)

            if stage_number is None:
                run_config_path = output_dir / f"train_run_{run_number:03d}.yaml"
            else:
                run_config_path = output_dir / f"train_run_{run_number:03d}_stage_{stage_number:02d}_{safe_name(stage_name)}.yaml"
            save_yaml(run_config_path, current_config)

            algorithm = current_config.get("model", {}).get("algorithm", "unknown")
            timesteps = current_config.get("train", {}).get("total_timesteps", "unknown")
            pretrained = current_config.get("model", {}).get("pretrained", "unknown")
            output_agent_name = current_config.get("train", {}).get("save_name", "unknown")
            warm_start_path = current_config.get("train", {}).get("warm_start_path", "")
            stage_text = "" if stage_number is None else f", stage={stage_number}/{stage_count} {stage_name}"

            print(
                f"[INFO] Starting training {run_number}/{run_count}{stage_text}: "
                f"algorithm={algorithm}, total_timesteps={timesteps}, "
                f"pretrained={pretrained}, warm_start_path={warm_start_path}, "
                f"output_agent_name={output_agent_name}, seed={current_config['train']['seed']}"
            )
            if item["output_model_path"] is not None:
                print(f"[INFO] Stage output model: {item['output_model_path']}")
            if args.dry_run:
                continue

            write_active_status(control_dir, {
                "status": "running",
                "run_number": run_number,
                "run_count": run_count,
                "stage_number": stage_number,
                "stage_count": stage_count,
                "stage_name": stage_name,
                "config_path": str(run_config_path),
                "stop_file": str(stop_file),
                "algorithm": algorithm,
                "total_timesteps": timesteps,
                "output_agent_name": output_agent_name,
                "output_model_path": str(item["output_model_path"]) if item["output_model_path"] else None,
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            })

            try:
                returncode, skipped = run_training(run_config_path, args.stop_timeout, stop_file)
            except KeyboardInterrupt:
                write_active_status(control_dir, {
                    "status": "aborted",
                    "run_number": run_number,
                    "run_count": run_count,
                    "stage_number": stage_number,
                    "stage_count": stage_count,
                    "stage_name": stage_name,
                    "config_path": str(run_config_path),
                    "stop_file": str(stop_file),
                    "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                })
                print("[INFO] Multi-training aborted by user.")
                return 130

            if skipped:
                write_active_status(control_dir, {
                    "status": "stopped_current_starting_next",
                    "run_number": run_number,
                    "run_count": run_count,
                    "stage_number": stage_number,
                    "stage_count": stage_count,
                    "stage_name": stage_name,
                    "config_path": str(run_config_path),
                    "stop_file": str(stop_file),
                    "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                })
                print(
                    f"[INFO] Training {run_number}/{run_count}{stage_text} was stopped by user. "
                    "Starting next training."
                )
                skip_remaining_stages = True
                break

            if returncode != 0:
                write_active_status(control_dir, {
                    "status": "failed",
                    "run_number": run_number,
                    "run_count": run_count,
                    "stage_number": stage_number,
                    "stage_count": stage_count,
                    "stage_name": stage_name,
                    "config_path": str(run_config_path),
                    "stop_file": str(stop_file),
                    "returncode": returncode,
                    "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                })
                print(
                    f"[ERROR] Training {run_number}/{run_count}{stage_text} failed with return code "
                    f"{returncode}."
                )
                return returncode

            write_active_status(control_dir, {
                "status": "completed_stage" if stage_number is not None else "completed_run",
                "run_number": run_number,
                "run_count": run_count,
                "stage_number": stage_number,
                "stage_count": stage_count,
                "stage_name": stage_name,
                "config_path": str(run_config_path),
                "stop_file": str(stop_file),
                "returncode": returncode,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            })

        if skip_remaining_stages:
            continue

        if stages and not args.dry_run:
            write_active_status(control_dir, {
                "status": "completed_run",
                "run_number": run_number,
                "run_count": run_count,
                "stage_count": len(stages),
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            })

    if args.dry_run:
        write_active_status(control_dir, {
            "status": "dry_run_completed",
            "run_count": run_count,
            "base_config": str(base_config_path),
            "multi_config": str(multi_config_path),
            "generated_config_dir": str(output_dir),
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        })
        print("[INFO] Dry run completed. No trainings were launched.")
    else:
        write_active_status(control_dir, {
            "status": "all_completed",
            "run_count": run_count,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        })
        print("[INFO] All trainings completed.")
    return 0


