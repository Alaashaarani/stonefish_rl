"""Evaluate or compare saved Stonefish RL policies."""
import argparse
import time
from pathlib import Path
import numpy as np
from library.configuration import DEFAULT_CONFIG
from library.evaluation import (
    load_config, build_model_specs, chunks, run_parallel, run_sequential,
    write_current_run_checklist, write_top_model_tables, write_error_tables,
    print_summary_table,
)
from library.evaluation_results import result_rows

def main():
    parser = argparse.ArgumentParser(description="Evaluate or compare Stonefish RL task models.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML. Defaults to include/parameters/stonefish_rl.yaml (evaluation section).",
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
        help="Directory where CSV/JSON results will be written. Default: evaluation_results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-step printing.",
    )
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG
    config = load_config(config_path)

    compare_enabled, specs, num_episodes, step_per_print = build_model_specs(config)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)


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
    results = []
    batch_size = int(compare_cfg.get("batch_size", 20)) if compare_enabled else len(specs)
    total_batches = max(1, int(np.ceil(len(specs) / max(1, batch_size))))
    for batch_number, spec_batch in chunks(specs, batch_size):
        if compare_enabled:
            print(
                f"\n[INFO] Evaluation batch {batch_number}/{total_batches}: "
                f"{len(spec_batch)} model(s)"
            )
        if use_parallel:
            results.extend(run_parallel(config, spec_batch, num_episodes, step_per_print, output_dir, quiet=args.quiet))
        else:
            results.extend(run_sequential(config, spec_batch, num_episodes, step_per_print, output_dir, quiet=args.quiet))
    elapsed = time.time() - start_time

    checklist = write_current_run_checklist(output_dir, results)
    summaries, all_episode_rows = result_rows(output_dir)
    write_top_model_tables(output_dir, summaries)
    errors = write_error_tables(output_dir, summaries)

    print_summary_table(summaries)
    print(f"\nCurrent run checklist saved to: {(output_dir / 'current_evaluation_checklist.csv').resolve()}")
    print(f"\nTop-10 tables saved to: {(output_dir / 'top_models').resolve()}")
    if errors:
        print(f"Evaluation errors saved to: {(output_dir / 'evaluation_errors.csv').resolve()}")
    print(f"\nEvaluation complete in {elapsed:.1f} s")
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
