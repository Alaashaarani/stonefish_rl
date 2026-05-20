"""Compatibility entry point for the football environment smoke test."""

import runpy


if __name__ == "__main__":
    runpy.run_module("stonefish_rl.scripts.test_football_env", run_name="__main__")

