import argparse
import copy
import csv
import json
import os
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

from stonefish_rl.envs.docking_env import dsEnv
from stonefish_rl.utils.utils import resolve_model_path, resolve_path


SUPPORTED_ALGORITHMS = {"SAC", "PPO", "TD3", "ONNX"}
SB3_ALGORITHMS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(resolve_path(config_path), "r") as f:
        return yaml.safe_load(f)


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
        "model_id", "model_name", "algorithm", "model_path",
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


def as_float(value):
    try:
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    except Exception:
        pass
    return None


def flatten_final_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract final scalar metrics from env info without assuming all keys exist."""
    out = {}
    if not isinstance(info, dict):
        return out

    for key, value in info.items():
        scalar = as_float(value)
        if scalar is not None:
            out[f"final_{key}"] = scalar
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float64).flatten()
            if arr.size > 0 and np.all(np.isfinite(arr)):
                for i, v in enumerate(arr):
                    out[f"final_{key}_{i}"] = float(v)

                # Common useful interpretation for error = [x, y, z, yaw]
                if key == "error" and arr.size >= 4:
                    out["final_x_error"] = float(arr[0])
                    out["final_y_error"] = float(arr[1])
                    out["final_z_error"] = float(arr[2])
                    out["final_yaw_error"] = float(arr[3])
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


def load_model(algorithm: str, model_path: str, env):
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
    quiet: bool = False,
) -> Dict[str, Any]:
    """Evaluate one model and return summary + per-episode metrics.

    Each worker creates its own simulator through dsEnv. For parallel runs, pass a
    unique base_port to avoid socket conflicts between Stonefish instances.
    """

    algorithm = algorithm.upper()
    model_name = Path(model_path).stem
    output_dir = Path(output_dir)

    local_config = copy.deepcopy(config)
    local_config.setdefault("evaluate", {})["algorithm"] = algorithm
    local_config.setdefault("evaluate", {})["model_path"] = model_path

    if base_port is not None:
        local_config.setdefault("env", {})["base_port"] = int(base_port)

    # In parallel mode, multiple GUI instances often conflict. The caller can set this.
    compare_cfg = local_config.get("compare", {}) or {}
    if compare_cfg.get("force_headless_parallel", False):
        local_config.setdefault("sim", {})["graphical_interface"] = False

    base_seed = evaluation_seed(local_config)

    env = None
    episode_rows = []
    error_message = None

    try:
        env = dsEnv(env_rank, local_config)
        model = load_model(algorithm, model_path, env)

        if not quiet:
            print(f"\n[Model {model_id}] Loaded {algorithm}: {model_path}")
            print(f"[Model {model_id}] base_port={local_config['env']['base_port']} env_rank={env_rank}")
            print(f"[Model {model_id}] evaluation seed base={base_seed}")

        for ep in range(num_episodes):
            episode_seed = None if base_seed is None else base_seed + ep
            obs, _ = env.reset(seed=episode_seed)
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
                "model_name": model_name,
                "algorithm": algorithm,
                "model_path": model_path,
                "episode": ep + 1,
                "success": bool(done),
                "truncated": bool(truncated),
                "total_reward": float(total_reward),
                "steps": int(step_counter),
                "collision_reward_sum": float(episode_collision_reward_sum),
                "seed": episode_seed,
            }
            row.update(flatten_final_info(final_info))
            row.update(summarize_info_accumulators(info_accum, step_counter))
            episode_rows.append(row)

            if not quiet:
                result_str = "SUCCESS" if done else "TIMEOUT/TRUNCATED"
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

    summary = summarize_model_rows(model_id, model_name, algorithm, model_path, episode_rows)
    if error_message is not None:
        summary["error"] = error_message

    # Write per-model result immediately. This is useful if one parallel worker crashes.
    per_model_dir = output_dir / f"model_{model_id:02d}_{model_name}"
    write_csv(per_model_dir / "episodes.csv", episode_rows)
    save_json(per_model_dir / "summary.json", summary)

    return {
        "summary": summary,
        "episodes": episode_rows,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def summarize_model_rows(model_id: int, model_name: str, algorithm: str, model_path: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    successes = [bool(r.get("success", False)) for r in rows]
    truncated = [bool(r.get("truncated", False)) for r in rows]
    rewards = [float(r["total_reward"]) for r in rows if "total_reward" in r]
    steps = [float(r["steps"]) for r in rows if "steps" in r]

    reward_mean, reward_std = mean_std(rewards)
    steps_mean, steps_std = mean_std(steps)

    summary = {
        "model_id": model_id,
        "model_name": model_name,
        "algorithm": algorithm,
        "model_path": model_path,
        "num_episodes": n,
        "success_count": int(sum(successes)),
        "success_rate": float(sum(successes) / n) if n else 0.0,
        "timeout_count": int(sum(truncated)),
        "mean_reward": reward_mean,
        "std_reward": reward_std,
        "mean_steps": steps_mean,
        "std_steps": steps_std,
    }

    # Add means of common final metrics if available.
    for key in [
        "final_xy_error",
        "final_z_error",
        "final_yaw_error",
        "mean_acceleration",
        "collision_reward_sum",
    ]:
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
        "id", "model", "alg", "episodes", "success", "succ_rate",
        "mean_reward", "std_reward", "mean_steps", "collision_sum_mean", "collision_sum_std", "collision_sum_total", "final_xy", "final_z", "final_yaw",
    ]

    rows = []
    for s in summaries:
        rows.append([
            s.get("model_id", ""),
            s.get("model_name", ""),
            s.get("algorithm", ""),
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
            specs.append({
                "model_id": i,
                "algorithm": str(alg).upper(),
                "model_path": str(path),
            })
        return True, specs, num_episodes, step_per_print

    # Fallback: original single-model evaluation mode.
    specs = [{
        "model_id": 0,
        "algorithm": str(eval_cfg["algorithm"]).upper(),
        "model_path": str(eval_cfg["model_path"]),
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
            "quiet": quiet,
        })

    # spawn is safer when simulators and ML libraries are involved.
    ctx = get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(_parallel_worker, worker_args)


def main():
    parser = argparse.ArgumentParser(description="Evaluate or compare Stonefish RL docking models.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to evaluation YAML. If omitted, uses include/parameters/evaluation_param.yaml.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate comparison models in parallel subprocesses. Requires unique ports and enough CPU/GPU resources.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential evaluation even if compare.parallel is true in YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where CSV/JSON results will be written.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-step printing.",
    )
    args = parser.parse_args()

    config_path = args.config or resolve_path("include/parameters/evaluation_param.yaml")
    config = load_config(config_path)

    compare_enabled, specs, num_episodes, step_per_print = build_model_specs(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("evaluation_results") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "used_config.json", config)

    compare_cfg = config.get("compare", {}) or {}
    yaml_parallel = bool(compare_cfg.get("parallel", False))
    use_parallel = compare_enabled and (args.parallel or yaml_parallel) and not args.sequential

    if use_parallel and config.get("sim", {}).get("graphical_interface", False):
        print(
            "WARNING: parallel evaluation with graphical_interface=True may fail because multiple simulator GUIs "
            "can conflict. Add compare.force_headless_parallel: True to the YAML, or run with --sequential."
        )

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'compare' if compare_enabled else 'single'} | parallel={use_parallel}")
    print(f"Models: {len(specs)} | Episodes/model: {num_episodes}")

    start_time = time.time()
    if use_parallel:
        results = run_parallel(config, specs, num_episodes, step_per_print, output_dir, quiet=args.quiet)
    else:
        results = run_sequential(config, specs, num_episodes, step_per_print, output_dir, quiet=args.quiet)
    elapsed = time.time() - start_time

    all_episode_rows = []
    summaries = []
    for result in results:
        summaries.append(result["summary"])
        all_episode_rows.extend(result["episodes"])

    write_csv(output_dir / "episodes_all_models.csv", all_episode_rows)
    write_csv(output_dir / "summary.csv", summaries)
    save_json(output_dir / "summary.json", summaries)

    print_summary_table(summaries)
    print(f"\nEvaluation complete in {elapsed:.1f} s")
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
