"""Factory helpers for selecting Stonefish RL environments by name."""

from __future__ import annotations

from typing import Any, Dict, Type

from stonefish_rl.envs.docking_env import dsEnv
from stonefish_rl.envs.football_env import FootballEnv


ENV_CLASSES: Dict[str, Type] = {
    "docking": dsEnv,
    "ds": dsEnv,
    "docking_ds": dsEnv,
    "football": FootballEnv,
}


def make_env_class(env_type: str):
    """Return the environment class registered for `env_type`."""
    key = str(env_type or "docking").lower()
    if key not in ENV_CLASSES:
        valid = ", ".join(sorted(ENV_CLASSES))
        raise ValueError(f"Unknown env_type '{env_type}'. Available: {valid}")
    return ENV_CLASSES[key]


def make_env_instance(env_type: str, rank: int, config: Dict[str, Any]):
    """Instantiate an environment by name."""
    return make_env_class(env_type)(rank, config)
