"""Factory helpers for selecting Stonefish RL environments by name."""

from __future__ import annotations

from typing import Any, Dict, Type
import importlib

from library.docking_env import dsEnv
from library.football_env import FootballEnv
from library.pid_tuning_env import PIDTuningEnv
from library.pid_rl_blend_env import PIDRLBlendEnv
from library.reach_target_env import ReachTargetEnv
from library.team_football_env import TeamFootballSoloEnv


ENV_CLASSES: Dict[str, Type] = {
    "docking": dsEnv,
    "ds": dsEnv,
    "docking_ds": dsEnv,
    "football": FootballEnv,
    "pid_tuning": PIDTuningEnv,
    "pid_rl_blend": PIDRLBlendEnv,
    "reach_target": ReachTargetEnv,
    "team_football_solo": TeamFootballSoloEnv,
}


def make_env_class(env_type: str, entry_point=None):
    """Return the environment class registered for `env_type`."""
    if entry_point:
        if not isinstance(entry_point, str) or entry_point.count(':') != 1:
            raise ValueError('env.entry_point must be an import path: package.module:ClassName')
        module_name, class_name = entry_point.split(':')
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ValueError('Cannot import env.entry_point %r: %s. Make the package available on PYTHONPATH.' %
                             (entry_point, exc)) from exc
        import gymnasium as gym
        if not isinstance(cls, type) or not issubclass(cls, gym.Env):
            raise ValueError('env.entry_point must name a Gymnasium Env subclass.')
        return cls
    key = str(env_type or "docking").lower()
    if key not in ENV_CLASSES:
        valid = ", ".join(sorted(ENV_CLASSES))
        raise ValueError(f"Unknown env_type '{env_type}'. Available: {valid}")
    return ENV_CLASSES[key]


def make_env_instance(env_type: str, rank: int, config: Dict[str, Any]):
    """Instantiate an environment by name."""
    return make_env_class(env_type, config.get('env', {}).get('entry_point'))(rank, config)


def reload_docking_environment():
    """Reload saved docking edits for the test GUI, after its old env is closed."""
    importlib.invalidate_caches()
    # Dependencies first, so from-imports bind to the updated definitions.
    for name in ("library.pose_frames", "library.env_base",
                 "library.docking_env",
                 "library.pid_tuning_env", "library.pid_rl_blend_env"):
        importlib.reload(importlib.import_module(name))
    global dsEnv, PIDTuningEnv, PIDRLBlendEnv
    dsEnv = importlib.import_module("library.docking_env").dsEnv
    PIDTuningEnv = importlib.import_module("library.pid_tuning_env").PIDTuningEnv
    for key in ("docking", "ds", "docking_ds"):
        ENV_CLASSES[key] = dsEnv
    ENV_CLASSES["pid_tuning"] = PIDTuningEnv
    PIDRLBlendEnv = importlib.import_module("library.pid_rl_blend_env").PIDRLBlendEnv
    ENV_CLASSES["pid_rl_blend"] = PIDRLBlendEnv
