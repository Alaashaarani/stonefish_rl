"""Blend fixed PID/frozen PPO commands, optionally with an upward retry command."""

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO

from library.docking_env import dsEnv
from library.pid_docking import PIDDockingPolicy
from library.common import resolve_model_path, resolve_path


def load_frozen_controller(config):
    """Validate the configured layout before starting a simulator."""
    blend = config.get("pid_rl_blend", {})
    if not blend.get("model_path"):
        raise ValueError("pid_rl_blend.model_path must specify a frozen PPO checkpoint")
    tcm = np.asarray(config.get("action", {}).get("tcm"), dtype=np.float32)
    if (not config.get("action", {}).get("force_6Dof") or tcm.ndim != 2
            or tcm.shape[0] not in (4, 6) or not np.all(np.isfinite(tcm))):
        raise ValueError("PID/RL blending requires a finite four- or six-row force TCM")
    with open(resolve_path(config["env"]["action_config"])) as stream:
        actuator_count = len(yaml.safe_load(stream)["action_config"]["specs"])
    if tcm.shape[1] != actuator_count:
        raise ValueError("TCM columns must match the actuator YAML")
    with open(resolve_path(config["env"]["state_config"])) as stream:
        state_count = len(yaml.safe_load(stream)["state_config"]["specs"])
    env = config["env"]
    frame_size = state_count + (env.get("yaw_observation", "angle") == "cos_sin")
    if env.get("observe_actions", False):
        frame_size += tcm.shape[0]
    obs_size = frame_size * (1 + max(0, int(env.get("history_length", 0))))
    if config["action"].get("use_thruster_health", False):
        obs_size += actuator_count
    model = PPO.load(resolve_model_path(blend["model_path"]), device="cpu")
    if model.observation_space.shape != (obs_size,):
        raise ValueError(f"Frozen PPO observation mismatch: checkpoint {model.observation_space.shape}, configured {(obs_size,)}")
    if (not isinstance(model.action_space, gym.spaces.Box)
            or model.action_space.shape != (tcm.shape[0],)
            or not np.all(model.action_space.low == -1)
            or not np.all(model.action_space.high == 1)):
        raise ValueError("Frozen PPO must output normalized vehicle commands matching the TCM rows")
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    return model


class PIDRLBlendEnv(gym.Wrapper):
    """Unchanged docking observations, with K1 and optional upward factor K2."""

    def __init__(self, rank, config):
        blend = config.get("pid_rl_blend", {})
        self.enable_upward = blend.get("enable_upward", False)
        if not isinstance(self.enable_upward, bool):
            raise ValueError("pid_rl_blend.enable_upward must be a boolean")
        self.upward_z = float(blend.get("upward_z", -1.0))
        if not np.isfinite(self.upward_z) or not -1 <= self.upward_z < 0:
            raise ValueError("pid_rl_blend.upward_z must be in [-1, 0) for body -Z ascent")
        self.rl_policy = load_frozen_controller(config)
        env = dsEnv(rank, config)
        super().__init__(env)
        try:
            if self.rl_policy.observation_space.shape != env.observation_space.shape:
                raise ValueError("Frozen PPO observation space does not match the docking environment")
            if self.rl_policy.action_space != env.action_space:
                raise ValueError("Frozen PPO action space does not match the docking environment")
            self.pid_policy = PIDDockingPolicy(env, config.get("pid", {}))
        except Exception:
            env.close()
            raise
        self.action_size = 2 if self.enable_upward else 1
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_size,), dtype=np.float32)
        self.upward_command = np.zeros(env.action_space.shape, dtype=np.float32)
        self.upward_command[2] = self.upward_z
        self._observation = None

    def reset(self, *, seed=None, options=None):
        self._observation = None
        self.pid_policy.reset()
        observation, info = self.env.reset(seed=seed, options=options)
        self._observation = observation.copy()
        return observation, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Reset PIDRLBlendEnv before stepping")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(f"Blend action must contain {self.action_size} finite factor(s)")
        k = float(np.clip(action[0], -1, 1))
        # Both controllers see the same previous observation, even at the endpoints.
        pid_action, _ = self.pid_policy.predict(self._observation)
        rl_action, _ = self.rl_policy.predict(self._observation, deterministic=True)
        commands = []
        for command in (pid_action, rl_action):
            command = np.asarray(command, dtype=np.float32)
            if command.shape != self.env.action_space.shape or not np.all(np.isfinite(command)):
                raise ValueError("Controller produced invalid vehicle commands")
            commands.append(np.clip(command, self.env.action_space.low, self.env.action_space.high))
        pid_action, rl_action = commands
        pid_weight = 0.5 + 0.5 * k
        pid_rl_command = (pid_weight * pid_action + (1 - pid_weight) * rl_action).astype(np.float32)
        k2 = float(np.clip(action[1], -1, 1)) if self.enable_upward else -1.0
        upward_weight = 0.5 + 0.5 * k2
        blend_weight = 1 - upward_weight
        # K2 controls only body Z; preserve K1's lateral and angular commands.
        command = pid_rl_command.copy()
        command[2] = blend_weight * pid_rl_command[2] + upward_weight * self.upward_z
        # TCM allocation, health scaling, rewards and command history occur once below.
        observation, reward, terminated, truncated, info = self.env.step(command)
        self._observation = None if terminated or truncated else observation.copy()
        info = dict(info)
        info.update(blend_k=k, blend_k1=k, blend_k2=k2,
                    pid_weight=pid_weight, rl_weight=1 - pid_weight,
                    blend_weight=blend_weight, upward_weight=upward_weight,
                    pid_command=pid_action.tolist(), rl_command=rl_action.tolist(),
                    pid_rl_command=pid_rl_command.tolist(), upward_command=self.upward_command.tolist(),
                    blended_command=command.tolist())
        return observation, reward, terminated, truncated, info
