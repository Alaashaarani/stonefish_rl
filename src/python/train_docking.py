"""Compatibility entry point for single-run docking training."""

import runpy


if __name__ == "__main__":
    runpy.run_module("stonefish_rl.training.train_docking", run_name="__main__")

