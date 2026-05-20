"""Compatibility entry point for the docking environment smoke test."""

import runpy


if __name__ == "__main__":
    runpy.run_module("stonefish_rl.scripts.test_docking_env", run_name="__main__")

