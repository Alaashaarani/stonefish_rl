import argparse
import copy
import datetime as dt
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MULTI_CONFIG = REPO_ROOT / "include/parameters/multi_train.yaml"
DEFAULT_BASE_CONFIG = REPO_ROOT / "include/parameters/train_param.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "include/parameters/generated_multi_train"


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)


def get_first(config, keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_reward_vector(value):
    return (
        isinstance(value, list)
        and len(value) == 6
        and all(is_number(item) for item in value)
    )


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


def build_run_config(base_config, multi_config, run_index, run_count):
    config = copy.deepcopy(base_config)

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
                "'reward_weights' entries must be reward vectors with exactly 6 numbers."
            )
        set_nested(config, "reward", "weights", weights)

    if max_iteration is not None:
        config.setdefault("train", {})
        config["train"]["total_timesteps"] = value_for_run(
            max_iteration,
            run_index,
            run_count,
            "max_iteration",
        )

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
        config["model"]["pretrained"] = value_for_run(
            pre_trained,
            run_index,
            run_count,
            "pre_trained",
        )

    if pretrained is not None:
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

    return config


def run_training(config_path):
    command = [
        sys.executable,
        "-m",
        "stonefish_rl.training.train_docking",
        str(config_path),
    ]
    return subprocess.run(command, cwd=REPO_ROOT / "src/python")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch train_docking.py multiple times from a multi-train YAML."
    )
    parser.add_argument(
        "multi_config",
        nargs="?",
        default=str(DEFAULT_MULTI_CONFIG),
        help="Path to multi_train.yaml.",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Override the base training YAML. Defaults to base_config in multi_train.yaml, then train_param.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated per-run YAML files are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate per-run YAML files and print the planned trainings without launching train_docking.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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

    base_config = load_yaml(base_config_path)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() / timestamp
    print(f"[INFO] Base config: {base_config_path}")
    print(f"[INFO] Multi config: {multi_config_path}")
    print(f"[INFO] Generated configs: {output_dir}")

    for index in range(run_count):
        run_number = index + 1
        run_config = build_run_config(base_config, multi_config, index, run_count)
        run_config_path = output_dir / f"train_run_{run_number:03d}.yaml"
        save_yaml(run_config_path, run_config)

        algorithm = run_config.get("model", {}).get("algorithm", "unknown")
        timesteps = run_config.get("train", {}).get("total_timesteps", "unknown")
        pretrained = run_config.get("model", {}).get("pretrained", "unknown")
        output_agent_name = run_config.get("train", {}).get("save_name", "unknown")
        warm_start_path = run_config.get("train", {}).get("warm_start_path", "")

        print(
            f"[INFO] Starting training {run_number}/{run_count}: "
            f"algorithm={algorithm}, total_timesteps={timesteps}, "
            f"pretrained={pretrained}, warm_start_path={warm_start_path}, "
            f"output_agent_name={output_agent_name}"
        )
        if args.dry_run:
            continue

        result = run_training(run_config_path)
        if result.returncode != 0:
            print(
                f"[ERROR] Training {run_number}/{run_count} failed with return code "
                f"{result.returncode}."
            )
            return result.returncode

    if args.dry_run:
        print("[INFO] Dry run completed. No trainings were launched.")
    else:
        print("[INFO] All trainings completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
