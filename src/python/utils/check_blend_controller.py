"""Check the frozen controller and action dimensions without starting Stonefish."""

if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.


import argparse

from library.configuration import load_config
from library.pid_rl_blend_env import load_frozen_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="include/parameters/train_pid_rl_blend.yaml")
    args = parser.parse_args()
    config = load_config(args.config, "training")
    if config.get("env", {}).get("type") != "pid_rl_blend":
        parser.error("env.type must be pid_rl_blend")
    model = load_frozen_controller(config)
    action_size = 2 if config.get("pid_rl_blend", {}).get("enable_upward", False) else 1
    print(f"Frozen PPO: observations {model.observation_space.shape}, commands {model.action_space.shape}; blend action ({action_size},)")
    del model


if __name__ == "__main__":
    main()
