import argparse
import copy
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from multiprocessing import get_context
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import yaml
from stable_baselines3 import SAC, PPO, TD3

from library.docking_env import dsEnv
from library.env_factory import make_env_instance
from library.task_registry import episode_success, uses_docking_observations, supports_fixed_pid, task_type
from library.evaluation_results import save_result, result_rows
from library.pid_docking import PIDDockingPolicy, is_pid
from library.common import resolve_model_path, resolve_path
from library.configuration import load_config as load_workflow_config, DEFAULT_CONFIG


SUPPORTED_ALGORITHMS = {"SAC", "PPO", "TD3", "ONNX", "PID"}
SB3_ALGORITHMS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}


def load_config(config_path: str) -> Dict[str, Any]:
    return load_workflow_config(config_path, "evaluation")


def evaluation_seed(config: Dict[str, Any]) -> Optional[int]:
    """Return evaluation seed or None when explicitly disabled.

    `compare.seed` takes precedence for model comparisons. If no seed is
    configured, default to 0 so each model sees the same reset sequence:
    episode 1 uses seed 0, episode 2 uses seed 1, etc.
    """
    compare_cfg = config.get("compare", {}) or {}
    eval_cfg = config.get("evaluate", {}) or {}
    seed_value = compare_cfg.get("seed", eval_cfg.get("seed", 0))
    if seed_value is None:
        return None
    return int(seed_value)


def normalize_yaw_observation(value: Any, key: str = "yaw_observation") -> str:
    """Validate yaw observation mode used by the docking environment."""
    mode = str(value).strip().lower()
    if mode not in {"angle", "cos_sin"}:
        raise ValueError(f"{key} must be either 'angle' or 'cos_sin'. Got: {value!r}")
    return mode


def task_yaw_observation(config, value, key='yaw_observation'):
    """Only docking owns the angle/cos_sin convention."""
    return normalize_yaw_observation(value, key) if uses_docking_observations(config) else str(value)


def value_for_model(value: Any, index: int, model_count: int, key: str) -> Any:
    """Return scalar-or-per-model config values."""
    if isinstance(value, list):
        if len(value) == model_count:
            return value[index]
        if len(value) == 1:
            return value[0]
        raise ValueError(
            f"{key} must be a scalar, a one-item list, or a list with {model_count} items. "
            f"Got {len(value)} items."
        )
    return value


def normalize_history_length(value: Any, key: str = "history_length") -> int:
    try:
        history_length = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer. Got: {value!r}")
    if history_length < 0:
        raise ValueError(f"{key} must be >= 0. Got: {history_length}")
    return history_length


def normalize_history_gap(value: Any, key: str = "history_gap") -> float:
    try:
        history_gap = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number. Got: {value!r}")
    if history_gap < 0.0:
        raise ValueError(f"{key} must be >= 0. Got: {history_gap}")
    return history_gap


def yaml_specs_count(config_path: str, root_key: str) -> int:
    with open(resolve_path(config_path), "r") as f:
        data = yaml.safe_load(f) or {}
    return len(data.get(root_key, {}).get("specs", []) or [])


def env_base_observation_dim(config: Dict[str, Any], yaw_observation: str) -> int:
    env_cfg = config.get("env", {}) or {}
    action_cfg = config.get("action", {}) or {}
    raw_state_size = yaml_specs_count(env_cfg["state_config"], "state_config")
    action_spec_size = yaml_specs_count(env_cfg["action_config"], "action_config")

    if bool(action_cfg.get("force_6Dof", False)) and action_cfg.get("tcm") is not None:
        action_size = len(action_cfg.get("tcm") or [])
    else:
        action_size = action_spec_size

    policy_state_size = raw_state_size + (1 if yaw_observation == "cos_sin" else 0)
    if bool(env_cfg.get("observe_actions", False)):
        return policy_state_size + action_size
    return policy_state_size


def env_total_observation_dim(config: Dict[str, Any], yaw_observation: str, history_length: int) -> int:
    return env_base_observation_dim(config, yaw_observation) * (1 + history_length) + env_health_observation_dim(config)


def env_health_observation_dim(config: Dict[str, Any]) -> int:
    if config.get("action", {}).get("use_thruster_health", False):
        return yaml_specs_count(config["env"]["action_config"], "action_config")
    return 0


def model_observation_dim(algorithm: str, model_path: str) -> Tuple[Optional[int], Optional[str]]:
    algorithm = algorithm.upper()
    if algorithm not in SB3_ALGORITHMS:
        return None, "Observation auto-correction is only available for SB3 zip models."
    try:
        model = SB3_ALGORITHMS[algorithm].load(resolve_model_path(model_path))
        shape = getattr(model.observation_space, "shape", None)
        if shape is None:
            return None, "Model has no observation_space.shape metadata."
        return int(np.prod(shape)), None
    except Exception as exc:
        return None, f"Could not inspect model observation shape: {exc}"


def auto_correct_observation_config(
    config: Dict[str, Any],
    algorithm: str,
    model_path: str,
    requested_yaw_observation: str,
    requested_history_length: int,
) -> Dict[str, Any]:
    """Correct yaw/history settings to match a saved SB3 model when possible."""
    if not uses_docking_observations(config):
        return {'enabled': False, 'changed': False, 'model_observation_dim': None,
                'expected_observation_dim': None,
                'message': 'Task-defined observations: docking yaw/history auto-correction is disabled.'}
    if is_pid(algorithm, model_path):
        return {"enabled": False, "changed": False, "model_observation_dim": None,
                "message": "PID uses the configured docking observations."}
    result = {
        "enabled": bool((config.get("compare", {}) or {}).get("auto_correct_observation", True)),
        "changed": False,
        "model_observation_dim": None,
        "expected_observation_dim": env_total_observation_dim(
            config,
            requested_yaw_observation,
            requested_history_length,
        ),
        "message": "",
    }
    if not result["enabled"]:
        return result

    target_dim, inspect_error = model_observation_dim(algorithm, model_path)
    result["model_observation_dim"] = target_dim
    if target_dim is None:
        result["message"] = inspect_error or "Could not inspect model observation shape."
        return result

    if target_dim == result["expected_observation_dim"]:
        result["message"] = "Observation config already matches model."
        return result

    yaw_candidates = [requested_yaw_observation]
    alternate_yaw = "cos_sin" if requested_yaw_observation == "angle" else "angle"
    yaw_candidates.append(alternate_yaw)

    matches = []
    history_dim = target_dim - env_health_observation_dim(config)
    for yaw_mode in yaw_candidates:
        base_dim = env_base_observation_dim(config, yaw_mode)
        if base_dim <= 0 or history_dim % base_dim != 0:
            continue
        inferred_history = history_dim // base_dim - 1
        if inferred_history >= 0:
            matches.append((yaw_mode, int(inferred_history), base_dim))

    if not matches:
        result["message"] = (
            "No yaw/history combination matched model observation dim "
            f"{target_dim}; configured dim was {result['expected_observation_dim']}."
        )
        return result

    chosen_yaw, chosen_history, _ = matches[0]
    config.setdefault("env", {})["yaw_observation"] = chosen_yaw
    config["env"]["history_length"] = chosen_history
    result["changed"] = True
    result["expected_observation_dim"] = target_dim
    result["message"] = (
        "Auto-corrected observation config: "
        f"yaw_observation {requested_yaw_observation!r} -> {chosen_yaw!r}, "
        f"history_length {requested_history_length} -> {chosen_history}."
    )
    return result


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Union of keys, preserving a useful front order first.
    front_keys = [
        "model_id", "evaluation_name", "evaluation_tag", "model_name", "algorithm", "model_path",
        "yaw_observation", "history_length", "history_gap",
        "episode", "success", "truncated", "total_reward", "steps",
        "final_xy_error", "final_z_error", "final_yaw_error",
        "collision_reward_sum",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    ordered_keys = [k for k in front_keys if k in all_keys]
    ordered_keys += sorted(k for k in all_keys if k not in ordered_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def chunks(items: List[Any], size: int):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield start // size + 1, items[start:start + size]


def safe_model_folder_name(model_name: str) -> str:
    """Create a stable folder name so re-evaluating a model replaces its files."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("._")
    return safe or "model"


def normalize_evaluation_tag(tag: Optional[Any]) -> str:
    if tag is None:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag).strip()).strip("._")


def evaluation_name_for_model(model_name: str, evaluation_tag: Optional[Any]) -> Tuple[str, str]:
    tag = normalize_evaluation_tag(evaluation_tag)
    if not tag:
        return str(model_name), ""
    return f"{model_name}_{tag}", tag


def folder_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def rebuild_root_indexes(output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Export optional CSV indexes from the independent evaluation records."""
    summaries, episodes = result_rows(output_dir)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "episodes_all_models.csv", episodes)
    save_json(output_dir / "summary.json", summaries)
    return summaries, episodes


def finite_float(value, default=float("-inf")):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def ranking_row(row: Dict[str, Any], rank: int, metric_name: str, metric_value: float) -> Dict[str, Any]:
    return {
        "rank": rank,
        "metric": metric_name,
        "metric_value": metric_value,
        "evaluation_name": row.get("evaluation_name", row.get("model_name", "")),
        "model_name": row.get("model_name", ""),
        "algorithm": row.get("algorithm", ""),
        "model_path": row.get("model_path", ""),
        "success_rate": row.get("success_rate", ""),
        "mean_reward": row.get("mean_reward", ""),
        "collision_reward_sum_mean": row.get("collision_reward_sum_mean", ""),
        "collision_reward_sum_total": row.get("collision_reward_sum_total", ""),
        "mean_steps": row.get("mean_steps", ""),
        "num_episodes": row.get("num_episodes", ""),
    }


def top_by_metric(summaries: List[Dict[str, Any]], metric_key: str, metric_name: str, reverse=True) -> List[Dict[str, Any]]:
    sorted_rows = sorted(
        summaries,
        key=lambda row: finite_float(row.get(metric_key)),
        reverse=reverse,
    )
    ranked = []
    for rank, row in enumerate(sorted_rows[:10], start=1):
        ranked.append(ranking_row(row, rank, metric_name, finite_float(row.get(metric_key))))
    return ranked


def write_top_model_tables(output_dir: Path, summaries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Write top-10 summary tables for quick model selection."""
    ranking_dir = output_dir / "top_models"
    ranking_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "success_rate": top_by_metric(summaries, "success_rate", "success_rate", reverse=True),
        "mean_reward": top_by_metric(summaries, "mean_reward", "mean_reward", reverse=True),
        # Collision rewards are penalties in this project, so larger/closer to zero is better.
        "collision_performance": top_by_metric(
            summaries,
            "collision_reward_sum_mean",
            "collision_reward_sum_mean",
            reverse=True,
        ),
    }

    write_csv(ranking_dir / "top_10_success_rate.csv", tables["success_rate"])
    write_csv(ranking_dir / "top_10_mean_reward.csv", tables["mean_reward"])
    write_csv(ranking_dir / "top_10_collision_performance.csv", tables["collision_performance"])
    save_json(ranking_dir / "top_10_tables.json", tables)
    return tables


def write_error_tables(output_dir: Path, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    errors = []
    for row in summaries:
        if not row.get("error"):
            continue
        errors.append({
            "evaluation_name": row.get("evaluation_name", row.get("model_name", "")),
            "model_name": row.get("model_name", ""),
            "algorithm": row.get("algorithm", ""),
            "model_path": row.get("model_path", ""),
            "yaw_observation": row.get("yaw_observation", ""),
            "history_length": row.get("history_length", ""),
            "history_gap": row.get("history_gap", ""),
            "auto_correction": row.get("auto_correction", ""),
            "error": row.get("error", ""),
        })
    write_csv(output_dir / "evaluation_errors.csv", errors)
    save_json(output_dir / "evaluation_errors.json", errors)
    return errors


def write_current_run_checklist(output_dir: Path, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        summary = result.get("summary", {}) or {}
        rows.append({
            "status": "failed" if summary.get("error") else "completed",
            "evaluation_name": summary.get("evaluation_name", summary.get("model_name", "")),
            "model_name": summary.get("model_name", ""),
            "algorithm": summary.get("algorithm", ""),
            "model_path": summary.get("model_path", ""),
            "num_episodes": summary.get("num_episodes", 0),
            "success_rate": summary.get("success_rate", ""),
            "mean_reward": summary.get("mean_reward", ""),
            "yaw_observation": summary.get("yaw_observation", ""),
            "history_length": summary.get("history_length", ""),
            "history_gap": summary.get("history_gap", ""),
            "auto_correction": summary.get("auto_correction", ""),
            "error": summary.get("error", ""),
        })
    write_csv(output_dir / "current_evaluation_checklist.csv", rows)
    save_json(output_dir / "current_evaluation_checklist.json", rows)
    return rows


def as_float(value):
    try:
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    except Exception:
        pass
    return None


def flatten_final_info(info: Dict[str, Any], docking=True) -> Dict[str, Any]:
    """Extract final scalar metrics from env info without assuming all keys exist."""
    out = {}
    if not isinstance(info, dict):
        return out

    for key, value in info.items():
        scalar = as_float(value)
        if scalar is not None:
            out[f"final_{key}"] = scalar
        elif isinstance(value, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(value, dtype=np.float64).flatten()
            except (TypeError, ValueError):
                continue  # Tasks may report labels or structured diagnostic data.
            if arr.size > 0 and np.all(np.isfinite(arr)):
                for i, v in enumerate(arr):
                    out[f"final_{key}_{i}"] = float(v)

                # Common useful interpretation for error = [x, y, z, yaw]
                if docking and key == "error" and arr.size >= 4:
                    out["final_x_error"] = float(arr[0])
                    out["final_y_error"] = float(arr[1])
                    out["final_z_error"] = float(arr[2])-0.91 #docking z offset
                    out["final_yaw_error"] =float(arr[3])   #docking yaw offset
                    out["final_xy_error"] = float(np.linalg.norm(arr[:2]))

    return out


def update_info_accumulators(info: Dict[str, Any], accum: Dict[str, float], count: int) -> None:
    """Accumulate numeric scalar info values for per-episode means.

    collision_reward is intentionally excluded here because it is more useful as
    an episode sum: averaging it over many zero-collision steps can hide severe
    collision events. The episode-level sum is computed explicitly in the
    evaluation loop and stored as collision_reward_sum.
    """
    if not isinstance(info, dict):
        return

    for key, value in info.items():
        if key == "collision_reward":
            continue
        scalar = as_float(value)
        if scalar is not None and np.isfinite(scalar):
            accum[key] = accum.get(key, 0.0) + scalar


def summarize_info_accumulators(accum: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {}
    return {f"mean_{key}": value / count for key, value in accum.items()}


def load_model(algorithm: str, model_path: str, env, config=None):
    if is_pid(algorithm, model_path):
        return PIDDockingPolicy(env, (config or {}).get("pid", {}))
    algorithm = algorithm.upper()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported: {sorted(SUPPORTED_ALGORITHMS)}")

    if algorithm in SB3_ALGORITHMS:
        return SB3_ALGORITHMS[algorithm].load(resolve_model_path(model_path), env=env)

    if algorithm == "ONNX":
        import openvino as ov
        core = ov.Core()
        ov_model = core.read_model(resolve_model_path(model_path))
        return core.compile_model(ov_model, "CPU")

    raise ValueError(f"Unsupported algorithm '{algorithm}'")


def predict_action(model, algorithm: str, obs: np.ndarray) -> np.ndarray:
    algorithm = algorithm.upper()
    if algorithm == "ONNX":
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        result = model(obs_batch)

        # OpenVINO may return a list, tuple, or dict depending on model export.
        if isinstance(result, dict):
            action = next(iter(result.values()))
        else:
            action = result[0]
        return np.asarray(action).flatten()

    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action).flatten()


def evaluate_one_model(
    model_id: int,
    algorithm: str,
    model_path: str,
    config: Dict[str, Any],
    num_episodes: int,
    step_per_print: int,
    output_dir: Union[str, Path],
    base_port: Optional[int] = None,
    env_rank: int = 0,
    yaw_observation: Optional[str] = None,
    history_length: Optional[int] = None,
    history_gap: Optional[float] = None,
    evaluation_tag: Optional[str] = None,
    quiet: bool = False,
    evaluation_name: Optional[str] = None,
    note: str = "",
) -> Dict[str, Any]:
    """Evaluate one model and return summary + per-episode metrics.

    Each worker creates its own simulator through dsEnv. For parallel runs, pass a
    unique base_port to avoid socket conflicts between Stonefish instances.
    """

    algorithm = "PID" if is_pid(algorithm, model_path) else algorithm.upper()
    model_name = "PID" if algorithm == "PID" else Path(model_path).stem
    default_name, evaluation_tag = evaluation_name_for_model(model_name, evaluation_tag)
    evaluation_name = evaluation_name or default_name
    output_dir = Path(output_dir)
    evaluated_at = datetime.now().isoformat(timespec="seconds")

    local_config = copy.deepcopy(config)
    local_config.setdefault("evaluate", {})["algorithm"] = algorithm
    local_config.setdefault("evaluate", {})["model_path"] = model_path
    if yaw_observation is not None:
        local_config.setdefault("env", {})["yaw_observation"] = task_yaw_observation(local_config,
            yaw_observation,
            f"model {model_id} yaw_observation",
        )
    yaw_observation = task_yaw_observation(local_config,
        local_config.setdefault("env", {}).get("yaw_observation", "angle"),
        f"model {model_id} env.yaw_observation",
    )
    local_config["env"]["yaw_observation"] = yaw_observation
    if history_length is not None:
        local_config.setdefault("env", {})["history_length"] = normalize_history_length(
            history_length,
            f"model {model_id} history_length",
        )
    history_length = normalize_history_length(
        local_config.setdefault("env", {}).get("history_length", 0),
        f"model {model_id} env.history_length",
    )
    local_config["env"]["history_length"] = history_length
    if history_gap is not None:
        local_config.setdefault("env", {})["history_gap"] = normalize_history_gap(
            history_gap,
            f"model {model_id} history_gap",
        )
    history_gap = normalize_history_gap(
        local_config.setdefault("env", {}).get("history_gap", 0.0),
        f"model {model_id} env.history_gap",
    )
    local_config["env"]["history_gap"] = history_gap

    if base_port is not None:
        local_config.setdefault("env", {})["base_port"] = int(base_port)

    # In parallel mode, multiple GUI instances often conflict. The caller can set this.
    compare_cfg = local_config.get("compare", {}) or {}
    if compare_cfg.get("force_headless_parallel", False):
        local_config.setdefault("sim", {})["graphical_interface"] = False

    correction = auto_correct_observation_config(
        local_config,
        algorithm,
        model_path,
        yaw_observation,
        history_length,
    )
    yaw_observation = task_yaw_observation(local_config,
        local_config["env"].get("yaw_observation", yaw_observation),
        f"model {model_id} corrected yaw_observation",
    )
    history_length = normalize_history_length(
        local_config["env"].get("history_length", history_length),
        f"model {model_id} corrected history_length",
    )
    if correction.get("message") and (correction.get("changed") or not quiet):
        print(f"[Model {model_id}] {correction['message']}")

    base_seed = evaluation_seed(local_config)

    env = None
    episode_rows = []
    error_message = None

    try:
        env_type = local_config.get("env", {}).get("type", "docking")
        if algorithm == "PID" and not supports_fixed_pid(local_config):
            raise ValueError('The fixed PID baseline supports docking only; use a trained model for %s.' % env_type)
        env = (dsEnv(env_rank, local_config)
               if env_type == "docking" and not local_config['env'].get('entry_point')
               else make_env_instance(env_type, env_rank, local_config))
        model = load_model(algorithm, model_path, env, local_config)

        if not quiet:
            print(f"\n[Model {model_id}] Loaded {algorithm}: {model_path}")
            print(f"[Model {model_id}] evaluation_name={evaluation_name}")
            print(f"[Model {model_id}] base_port={local_config['env']['base_port']} env_rank={env_rank}")
            print(f"[Model {model_id}] yaw_observation={yaw_observation}")
            print(f"[Model {model_id}] history_length={history_length}")
            print(f"[Model {model_id}] history_gap={history_gap}")
            print(f"[Model {model_id}] evaluation seed base={base_seed}")

        for ep in range(num_episodes):
            episode_seed = None if base_seed is None else base_seed + ep
            obs, _ = env.reset(seed=episode_seed)
            if algorithm == "PID":
                model.reset()
            done = False
            truncated = False
            total_reward = 0.0
            step_counter = 0
            info_accum = {}
            final_info = {}
            episode_collision_reward_sum = 0.0

            if not quiet:
                print(
                    f"\n[Model {model_id} | {model_name}] "
                    f"Starting episode {ep + 1}/{num_episodes} seed={episode_seed}"
                )

            while not (done or truncated):
                action = predict_action(model, algorithm, obs)
                obs, reward, done, truncated, info = env.step(action)

                total_reward += float(reward)
                step_counter += 1
                final_info = info if isinstance(info, dict) else {}
                if isinstance(final_info, dict):
                    collision_value = as_float(final_info.get("collision_reward", 0.0))
                    if collision_value is not None and np.isfinite(collision_value):
                        episode_collision_reward_sum += float(collision_value)
                update_info_accumulators(final_info, info_accum, step_counter)

                if step_per_print > 0 and step_counter % step_per_print == 0 and not quiet:
                    print(
                        f"[Model {model_id} | Ep {ep + 1}] "
                        f"step={step_counter} reward={float(reward):.3f} "
                        f"total={total_reward:.3f} action={np.round(action, 3)}"
                    )

            row = {
                "model_id": model_id,
                "evaluation_name": evaluation_name,
                "evaluation_tag": evaluation_tag,
                "model_name": model_name,
                "algorithm": algorithm,
                "model_path": model_path,
                "yaw_observation": yaw_observation,
                "history_length": history_length,
                "history_gap": history_gap,
                "auto_correction": correction.get("message", ""),
                "model_observation_dim": correction.get("model_observation_dim"),
                "evaluated_at": evaluated_at,
                "episode": ep + 1,
                "success": episode_success(final_info, done, legacy_termination=(
                    uses_docking_observations(local_config) or task_type(local_config) == 'football')),
                "terminated": bool(done),
                "task_type": task_type(local_config),
                "truncated": bool(truncated),
                "total_reward": float(total_reward),
                "steps": int(step_counter),
                "collision_reward_sum": float(episode_collision_reward_sum),
                "seed": episode_seed,
            }
            row.update(flatten_final_info(final_info, docking=uses_docking_observations(local_config)))
            row.update(summarize_info_accumulators(info_accum, step_counter))
            episode_rows.append(row)

            if not quiet:
                result_str = "SUCCESS" if row['success'] else "FAILED" if done else "TIMEOUT/TRUNCATED"
                print(
                    f"[Model {model_id} | Ep {ep + 1}] {result_str}: "
                    f"reward={total_reward:.3f}, steps={step_counter}"
                )

    except Exception as exc:
        error_message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"\n[Model {model_id}] ERROR while evaluating {model_path}:\n{error_message}", file=sys.stderr)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    summary = summarize_model_rows(
        model_id,
        model_name,
        algorithm,
        model_path,
        episode_rows,
        yaw_observation=yaw_observation,
        history_length=history_length,
        history_gap=history_gap,
        evaluation_name=evaluation_name,
        evaluation_tag=evaluation_tag,
    )
    summary["auto_correction"] = correction.get("message", "")
    summary['task_type'] = task_type(local_config)
    summary["model_observation_dim"] = correction.get("model_observation_dim")
    if error_message is not None:
        summary["error"] = error_message
    summary["evaluated_at"] = evaluated_at

    # Write per-model result immediately. This is useful if one parallel worker crashes.
    save_result(output_dir, summary, episode_rows, local_config, evaluation_name, note)

    return {
        "summary": summary,
        "episodes": episode_rows,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def summarize_model_rows(
    model_id: int,
    model_name: str,
    algorithm: str,
    model_path: str,
    rows: List[Dict[str, Any]],
    yaw_observation: Optional[str] = None,
    history_length: Optional[int] = None,
    history_gap: Optional[float] = None,
    evaluation_name: Optional[str] = None,
    evaluation_tag: Optional[str] = None,
) -> Dict[str, Any]:
    n = len(rows)
    successes = [bool(r.get("success", False)) for r in rows]
    truncated = [bool(r.get("truncated", False)) for r in rows]
    rewards = [float(r["total_reward"]) for r in rows if "total_reward" in r]
    steps = [float(r["steps"]) for r in rows if "steps" in r]

    reward_mean, reward_std = mean_std(rewards)
    steps_mean, steps_std = mean_std(steps)

    summary = {
        "model_id": model_id,
        "evaluation_name": evaluation_name or model_name,
        "evaluation_tag": evaluation_tag or "",
        "model_name": model_name,
        "algorithm": algorithm,
        "model_path": model_path,
        "yaw_observation": yaw_observation,
        "history_length": history_length,
        "history_gap": history_gap,
        "num_episodes": n,
        "success_count": int(sum(successes)),
        "success_rate": float(sum(successes) / n) if n else 0.0,
        "timeout_count": int(sum(truncated)),
        "failure_count": sum(bool(r.get('terminated')) and not bool(r.get('success')) for r in rows),
        "mean_reward": reward_mean,
        "std_reward": reward_std,
        "mean_steps": steps_mean,
        "std_steps": steps_std,
    }

    # Preserve task-specific numeric diagnostics without adding evaluator code.
    metric_keys = {key for row in rows for key in row
                   if key.startswith(('final_', 'mean_'))}
    for key in sorted(metric_keys | {'collision_reward_sum'}):
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        if vals:
            m, s = mean_std(vals)
            summary[f"{key}_mean"] = m
            summary[f"{key}_std"] = s

    collision_sums = [
        float(r["collision_reward_sum"])
        for r in rows
        if "collision_reward_sum" in r and np.isfinite(float(r["collision_reward_sum"]))
    ]
    if collision_sums:
        m, s = mean_std(collision_sums)
        summary["collision_reward_sum_mean"] = m
        summary["collision_reward_sum_std"] = s
        summary["collision_reward_sum_total"] = float(sum(collision_sums))

    return summary


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        print("No summaries to print.")
        return

    headers = [
        "id", "evaluation", "tag", "model", "alg", "yaw_obs", "hist", "hist_gap", "episodes", "success", "succ_rate",
        "mean_reward", "std_reward", "mean_steps", "collision_sum_mean", "collision_sum_std", "collision_sum_total", "final_xy", "final_z", "final_yaw",
    ]

    rows = []
    for s in summaries:
        rows.append([
            s.get("model_id", ""),
            s.get("evaluation_name", s.get("model_name", "")),
            s.get("evaluation_tag", ""),
            s.get("model_name", ""),
            s.get("algorithm", ""),
            s.get("yaw_observation", ""),
            s.get("history_length", ""),
            s.get("history_gap", ""),
            s.get("num_episodes", 0),
            s.get("success_count", 0),
            f"{100.0 * s.get('success_rate', 0.0):.1f}%",
            f"{s.get('mean_reward', float('nan')):.3f}",
            f"{s.get('std_reward', float('nan')):.3f}",
            f"{s.get('mean_steps', float('nan')):.1f}",
            f"{s.get('collision_reward_sum_mean', float('nan')):.3f}" if "collision_reward_sum_mean" in s else "",
            f"{s.get('collision_reward_sum_std', float('nan')):.3f}" if "collision_reward_sum_std" in s else "",
            f"{s.get('collision_reward_sum_total', float('nan')):.3f}" if "collision_reward_sum_total" in s else "",
            f"{s.get('final_xy_error_mean', float('nan')):.3f}" if "final_xy_error_mean" in s else "",
            f"{s.get('final_z_error_mean', float('nan')):.3f}" if "final_z_error_mean" in s else "",
            f"{s.get('final_yaw_error_mean', float('nan')):.3f}" if "final_yaw_error_mean" in s else "",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row_values):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row_values))

    print("\n" + fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def build_model_specs(config: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]], int, int]:
    compare_cfg = config.get("compare", {}) or {}
    eval_cfg = config.get("evaluate", {}) or {}

    compare_enabled = bool(compare_cfg.get("compare", False))

    if compare_enabled:
        algorithms = compare_cfg.get("algorithms", [])
        model_paths = compare_cfg.get("model_paths", [])
        yaw_observations = compare_cfg.get(
            "yaw_observation",
            config.get("env", {}).get("yaw_observation", "angle"),
        )
        history_lengths = compare_cfg.get(
            "history_length",
            config.get("env", {}).get("history_length", 0),
        )
        history_gaps = compare_cfg.get(
            "history_gap",
            config.get("env", {}).get("history_gap", 0.0),
        )
        evaluation_tags = compare_cfg.get(
            "evaluation_tag",
            compare_cfg.get("evaluation_tags", ""),
        )
        num_episodes = int(compare_cfg.get("num_episodes", eval_cfg.get("num_episodes", 1)))
        step_per_print = int(compare_cfg.get("step_per_print", eval_cfg.get("step_per_print", 0)))

        if len(algorithms) != len(model_paths):
            raise ValueError(
                "compare.algorithms and compare.model_paths must have the same length. "
                f"Got {len(algorithms)} algorithms and {len(model_paths)} paths."
            )

        num_models = int(compare_cfg.get("num_models", len(model_paths)))
        if num_models != len(model_paths):
            print(
                f"WARNING: compare.num_models={num_models}, but {len(model_paths)} model_paths were provided. "
                "Using the length of model_paths."
            )

        specs = []
        for i, (alg, path) in enumerate(zip(algorithms, model_paths)):
            yaw_observation = task_yaw_observation(config,
                value_for_model(yaw_observations, i, len(model_paths), "compare.yaw_observation"),
                f"compare.yaw_observation[{i}]",
            )
            specs.append({
                "model_id": i,
                "algorithm": str(alg).upper(),
                "model_path": str(path),
                "yaw_observation": yaw_observation,
                "history_length": normalize_history_length(
                    value_for_model(history_lengths, i, len(model_paths), "compare.history_length"),
                    f"compare.history_length[{i}]",
                ),
                "history_gap": normalize_history_gap(
                    value_for_model(history_gaps, i, len(model_paths), "compare.history_gap"),
                    f"compare.history_gap[{i}]",
                ),
                "evaluation_tag": value_for_model(
                    evaluation_tags,
                    i,
                    len(model_paths),
                    "compare.evaluation_tag",
                ),
                "evaluation_name": value_for_model(compare_cfg.get("evaluation_name", ""), i, len(model_paths), "compare.evaluation_name"),
                "note": value_for_model(compare_cfg.get("note", ""), i, len(model_paths), "compare.note"),
            })
        return True, specs, num_episodes, step_per_print

    # Fallback: original single-model evaluation mode.
    yaw_observation = task_yaw_observation(config,
        config.get("env", {}).get("yaw_observation", "angle"),
        "env.yaw_observation",
    )
    history_length = normalize_history_length(
        config.get("env", {}).get("history_length", 0),
        "env.history_length",
    )
    history_gap = normalize_history_gap(
        config.get("env", {}).get("history_gap", 0.0),
        "env.history_gap",
    )
    specs = [{
        "model_id": 0,
        "algorithm": str(eval_cfg["algorithm"]).upper(),
        "model_path": str(eval_cfg["model_path"]),
        "yaw_observation": yaw_observation,
        "history_length": history_length,
        "history_gap": history_gap,
        "evaluation_tag": eval_cfg.get("evaluation_tag", ""),
        "evaluation_name": eval_cfg.get("evaluation_name", ""),
        "note": eval_cfg.get("note", ""),
    }]
    num_episodes = int(eval_cfg.get("num_episodes", 1))
    step_per_print = int(eval_cfg.get("step_per_print", 0))
    return False, specs, num_episodes, step_per_print


def run_sequential(config, specs, num_episodes, step_per_print, output_dir, quiet=False):
    base_port0 = int(config.get("env", {}).get("base_port", 5595))
    port_stride = int((config.get("compare", {}) or {}).get("port_stride", 20))

    results = []
    for i, spec in enumerate(specs):
        result = evaluate_one_model(
            model_id=spec["model_id"],
            algorithm=spec["algorithm"],
            model_path=spec["model_path"],
            config=config,
            num_episodes=num_episodes,
            step_per_print=step_per_print,
            output_dir=output_dir,
            base_port=base_port0 + i * port_stride,
            env_rank=0,
            yaw_observation=spec.get("yaw_observation"),
            history_length=spec.get("history_length"),
            history_gap=spec.get("history_gap"),
            evaluation_tag=spec.get("evaluation_tag"),
            evaluation_name=spec.get("evaluation_name"),
            note=spec.get("note", ""),
            quiet=quiet,
        )
        results.append(result)
    return results


def _parallel_worker(args):
    return evaluate_one_model(**args)


def run_parallel(config, specs, num_episodes, step_per_print, output_dir, quiet=False):
    base_port0 = int(config.get("env", {}).get("base_port", 5595))
    compare_cfg = config.get("compare", {}) or {}
    port_stride = int(compare_cfg.get("port_stride", 20))
    max_workers = int(compare_cfg.get("max_workers", len(specs)))
    max_workers = max(1, min(max_workers, len(specs)))
    print(f"[INFO] Parallel evaluation pool: {max_workers} worker(s), {len(specs)} model(s)", flush=True)

    worker_args = []
    for i, spec in enumerate(specs):
        worker_args.append({
            "model_id": spec["model_id"],
            "algorithm": spec["algorithm"],
            "model_path": spec["model_path"],
            "config": config,
            "num_episodes": num_episodes,
            "step_per_print": step_per_print,
            "output_dir": str(output_dir),
            "base_port": base_port0 + i * port_stride,
            "env_rank": 0,
            "yaw_observation": spec.get("yaw_observation"),
            "history_length": spec.get("history_length"),
            "history_gap": spec.get("history_gap"),
            "evaluation_tag": spec.get("evaluation_tag"),
            "evaluation_name": spec.get("evaluation_name"),
            "note": spec.get("note", ""),
            "quiet": quiet,
        })

    # spawn is safer when simulators and ML libraries are involved.
    ctx = get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(_parallel_worker, worker_args)


