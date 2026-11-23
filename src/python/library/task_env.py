"""Reusable Gymnasium lifecycle for new Stonefish tasks.

Subclass ``StonefishTaskEnv`` and implement ``evaluate_transition``. Simulator
transport, action mapping, seeded resets, observation history and time limits
are shared; existing docking environments retain their original behavior.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Dict

import gymnasium as gym
import numpy as np

from library.env_base import EnvStonefishRLParallel, launch_stonefish_simulator
from library.task_registry import validate_task_reward
from library.domain_randomization import configure_domain_randomization


@dataclass
class TaskStep:
    reward: float
    terminated: bool = False
    success: bool = False
    info: Dict = field(default_factory=dict)


class StonefishTaskEnv(EnvStonefishRLParallel):
    """A task uses named raw states and returns a TaskStep for each transition."""

    def __init__(self, rank, config):
        self.config = deepcopy(config)
        self.task_config = self.config.get('task', {})
        cfg = self.config['env']
        self.state_path, self.act_path = cfg['state_config'], cfg['action_config']
        self.episode_duration = float(cfg['episode_duration'])
        self.rl_observation_freq = float(cfg['rl_observation_freq'])
        if not all(math.isfinite(x) and x > 0 for x in (self.episode_duration, self.rl_observation_freq)):
            raise ValueError('Episode duration and observation frequency must be positive and finite.')
        self.max_episode_steps = math.ceil(self.episode_duration * self.rl_observation_freq)
        self.history_length = cfg.get('history_length', 0)
        if type(self.history_length) is not int or self.history_length < 0:
            raise ValueError('env.history_length must be a nonnegative integer.')
        if cfg.get('history_gap', 0) != 0:
            raise ValueError('StonefishTaskEnv supports consecutive history only (history_gap: 0).')
        if config.get('action', {}).get('use_thruster_health', False):
            raise ValueError('New tasks need their own actuator-fault model; set use_thruster_health: false.')
        validate_task_reward(config)
        self.observe_actions = bool(cfg.get('observe_actions', False))
        self.use_thruster_health = False
        self.tcm = (np.asarray(config['action']['tcm'], dtype=np.float32)
                    if config.get('action', {}).get('force_6Dof', False) else None)
        if self.tcm is not None:
            if self.tcm.ndim != 2 or not self.tcm.size or not np.isfinite(self.tcm).all():
                raise ValueError('action.tcm must be a finite nonempty matrix.')
            self.tcm_inv = np.linalg.pinv(self.tcm)
        self._strict_transport = True
        self._communication_timeout_ms = cfg.get('communication_timeout_ms', 60000)
        if type(self._communication_timeout_ms) is not int or self._communication_timeout_ms <= 0:
            raise ValueError('env.communication_timeout_ms must be a positive integer.')
        self._domain_randomizer = configure_domain_randomization(config)
        self.process = None
        self.observation_history = []
        try:
            super().__init__(env_id=rank, base_port=cfg['base_port'])
            if not self.state_names or len(set(self.state_names)) != len(self.state_names):
                raise ValueError('State YAML needs unique, nonempty output names.')
            if not self.action_names:
                raise ValueError('Action YAML must define at least one actuator.')
            self.state_indices = {name: i for i, name in enumerate(self.state_names)}
            self.observation_scale = np.broadcast_to(
                np.asarray(cfg.get('observation_scale', 1.0), dtype=np.float32), (self.state_size,)).copy()
            if not np.isfinite(self.observation_scale).all() or (self.observation_scale <= 0).any():
                raise ValueError('env.observation_scale must contain positive finite values.')
            self.current_action = np.zeros(self.action_size, dtype=np.float32)
            self.step_count = 0
            self._needs_reset = True
            self.validate_task()
            self._validate_resets()
        except Exception:
            self.close()
            raise

    def validate_task(self):
        """Validate task-specific settings before the first simulator launch."""

    def state_values(self, names, state=None):
        """Select exact YAML output names; misspellings are errors, never zeros."""
        source = self.state if state is None else state
        try:
            return np.asarray([source[self.state_indices[name]] for name in names], dtype=np.float32)
        except KeyError as exc:
            raise ValueError('Unknown observation %s; available: %s' % (exc, ', '.join(self.state_names))) from exc

    def _validate_resets(self):
        resets = self.task_config.get('reset')
        if not isinstance(resets, list) or not resets:
            raise ValueError('task.reset must list the simulated robots to reset.')
        names = []
        for item in resets:
            if not isinstance(item, dict) or not isinstance(item.get('name'), str) or not item['name']:
                raise ValueError('Each task.reset entry needs a robot name.')
            names.append(item['name'])
            for key in ('position', 'rotation', 'position_min', 'position_max', 'current'):
                if key in item:
                    value = np.asarray(item[key], dtype=float)
                    if value.shape != (3,) or not np.isfinite(value).all():
                        raise ValueError('task.reset.%s must contain three finite numbers.' % key)
            if ('position_min' in item) != ('position_max' in item):
                raise ValueError('Supply both position_min and position_max for a randomized reset.')
            if 'position_min' in item and np.any(np.asarray(item['position_min']) > item['position_max']):
                raise ValueError('Reset position_min must not exceed position_max.')
        if len(set(names)) != len(names):
            raise ValueError('Reset robot names must be unique.')

    def build_reset_command(self):
        commands = []
        for source in self.task_config['reset']:
            item = deepcopy(source)
            if 'position_min' in item:
                item['position'] = self.np_random.uniform(item.pop('position_min'), item.pop('position_max')).tolist()
            item.setdefault('position', [0.0, 0.0, 0.0])
            item.setdefault('rotation', [0.0, 0.0, 0.0])
            commands.append(item)
        return commands

    def reset_task(self, options):
        """Optional episode counters; use self.np_random for task randomness."""

    def reset(self, *, seed=None, options=None):
        if self.process is None and self._domain_randomizer is None:
            self.process = launch_stonefish_simulator(self.env_id, self.config)
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.observation_history = []
        self.current_action = np.zeros(self.action_size, dtype=np.float32)
        self.reset_task(options or {})
        self._needs_reset = False
        return self.build_observation(), self._domain_randomization_info()

    def evaluate_transition(self, previous_state, action):
        """Return TaskStep; self.state contains the newly received raw state."""
        raise NotImplementedError('Implement evaluate_transition in your task subclass.')

    def build_observation(self):
        observation = self.state / self.observation_scale
        if self.observe_actions:
            observation = np.concatenate((observation, self.current_action))
        self.observation_history.append(observation.copy())
        self.observation_history = self.observation_history[-(self.history_length + 1):]
        result = np.concatenate(self.observation_history)
        self.observation = np.pad(result, (0, self.total_obs_size - result.size)).astype(np.float32)
        return self.observation

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded('Call reset() before stepping a new episode.')
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Action must be finite, within bounds, and shaped %s.' % (self.action_space.shape,))
        previous = self.state.copy()
        self.current_action = action.copy()
        try:
            # Use strict transport: no stale observation on a failed socket request.
            self._process_state_vector(self.send_command(self.build_command(action)))
            self.step_count += 1
            result = self.evaluate_transition(previous, action)
            if not isinstance(result, TaskStep) or not math.isfinite(result.reward):
                raise ValueError('evaluate_transition must return TaskStep with a finite reward.')
            if type(result.success) is not bool or type(result.terminated) is not bool:
                raise ValueError('TaskStep success and terminated must be bool values.')
            truncated = self.step_count >= self.max_episode_steps and not result.terminated
            self._needs_reset = result.terminated or truncated
            info = {**self._domain_randomization_info(), **result.info,
                    'is_success': result.success, 'step': self.step_count}
            return self.build_observation(), float(result.reward), result.terminated, truncated, info
        except Exception:
            self._needs_reset = True
            self.close()
            raise
