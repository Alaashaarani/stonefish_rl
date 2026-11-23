"""Docking with per-step policy actions that tune a persistent PID controller."""

import gymnasium as gym
import numpy as np

from library.docking_env import dsEnv
from library.pid_docking import PIDDockingPolicy


class PIDTuningEnv(gym.Wrapper):
    """Keep docking observations/rewards; replace commands with 16 PID parameters."""

    def __init__(self, rank, config):
        bounds = config.get("pid_tuning", {})
        lows, highs = [], []
        for key in ("kp", "ki", "kd", "offset"):
            block = bounds.get(key, {})
            low = np.asarray(block.get("low"), dtype=np.float64)
            high = np.asarray(block.get("high"), dtype=np.float64)
            if (low.shape != (4,) or high.shape != (4,)
                    or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
                    or np.any(low > high) or (key != "offset" and np.any(low < 0))):
                raise ValueError(f"pid_tuning.{key} requires finite low/high arrays of four values with low <= high; gains must be nonnegative")
            lows.extend(low)
            highs.extend(high)
        self.parameter_low = np.array(lows)
        self.parameter_high = np.array(highs)
        # Validate before launching a simulator.
        tcm = np.asarray(config.get("action", {}).get("tcm"))
        if not config.get("action", {}).get("force_6Dof") or tcm.ndim != 2 or tcm.shape[0] not in (4, 6):
            raise ValueError("PID tuning requires a four- or six-row force TCM")
        env = dsEnv(rank, config)
        super().__init__(env)
        try:
            self.pid_policy = PIDDockingPolicy(env, config.get("pid", {}))
        except Exception:
            env.close()
            raise
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.action_size = 16
        self._observation = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.pid_policy.reset()
        self._observation = observation.copy()
        return observation, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Reset PIDTuningEnv before stepping")
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.all(np.isfinite(action)):
            raise ValueError("PID tuning action must contain 16 finite values")
        fraction = (np.clip(action, -1, 1) + 1) / 2
        parameters = self.parameter_low + fraction * (self.parameter_high - self.parameter_low)
        pid = self.pid_policy.controller
        pid.kp, pid.ki, pid.kd, pid.offsets = parameters.reshape(4, 4).copy()
        command, _ = self.pid_policy.predict(self._observation)
        observation, reward, terminated, truncated, info = self.env.step(command)
        self._observation = observation.copy()
        info = dict(info)
        info["pid_parameters"] = parameters.tolist()
        info["pid_command"] = command.tolist()
        return observation, reward, terminated, truncated, info
