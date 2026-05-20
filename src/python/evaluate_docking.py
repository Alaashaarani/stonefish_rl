"""Compatibility entry point for single-model docking evaluation."""

import runpy


if __name__ == "__main__":
    runpy.run_module("stonefish_rl.evaluation.evaluate_docking", run_name="__main__")

