"""Read-only checks for task configurations; never construct an environment."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import yaml

from library.task_registry import validate_task_reward, task_type, supports_fixed_pid
from library.configuration import DEFAULT_CONFIG, load_config
from library.common import resolve_path, resolve_model_path


def check_config(config, mode='training'):
    errors, warnings = [], []
    if not isinstance(config, dict):
        return dict(ok=False, details={}, errors=['Configuration must be a mapping.'], warnings=[])
    for section in ('env', 'sim', 'action', 'model', 'train', 'log', 'evaluate', 'reward', 'domain_randomization'):
        if section in config and not isinstance(config[section], dict):
            errors.append('%s must be a YAML mapping.' % section)
    if errors:
        return dict(ok=False, details={}, errors=errors, warnings=[])
    details = {'task': task_type(config), 'mode': mode}

    def attempt(label, callback):
        try:
            return callback()
        except (ValueError, TypeError, KeyError, OSError, ImportError, yaml.YAMLError) as exc:
            errors.append('%s: %s' % (label, exc))

    def number(value, label, integer=False, minimum=0, inclusive=False):
        valid = (type(value) is int if integer else isinstance(value, (int, float)) and not isinstance(value, bool))
        if not valid or not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
            raise ValueError('%s must be %s %s %s.' % (
                label, 'an integer' if integer else 'a finite number', '>=' if inclusive else '>', minimum))
        return value

    def environment():
        from library.env_factory import make_env_class
        cfg = config['env']
        cls = make_env_class(cfg.get('type', 'docking'), cfg.get('entry_point'))
        details['environment_class'] = '%s:%s' % (cls.__module__, cls.__name__)
        instances = number(cfg.get('instances', 1), 'env.instances', integer=True)
        port = number(cfg['base_port'], 'env.base_port', integer=True, minimum=1024, inclusive=True)
        number(cfg['episode_duration'], 'env.episode_duration')
        number(cfg['rl_observation_freq'], 'env.rl_observation_freq')
        number(cfg.get('history_length', 0), 'env.history_length', integer=True, inclusive=True)
        if port + instances > 65536:
            raise ValueError('Simulator ports exceed 65535.')
        details.update(instances=instances, episode_steps=math.ceil(cfg['episode_duration'] * cfg['rl_observation_freq']))

    attempt('Environment', environment)
    attempt('Rewards', lambda: validate_task_reward(config))

    def file_path(section, key, directory=False):
        value = config.get(section, {}).get(key)
        if not isinstance(value, str) or not value:
            raise ValueError('%s.%s must specify a path.' % (section, key))
        path = Path(resolve_path(value))
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError('Not found: %s' % path)
        return path

    attempt('Scene', lambda: file_path('sim', 'scene_path'))
    attempt('Resources', lambda: file_path('sim', 'resources_path', directory=True))
    specs = {}
    for key, root in (('state_config', 'state_config'), ('action_config', 'action_config')):
        def read_specs(key=key, root=root):
            path = file_path('env', key)
            document = yaml.safe_load(path.read_text())
            rows = document[root]['specs']
            if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
                raise ValueError('%s.specs must be a nonempty list of mappings.' % root)
            names = [row.get('output_name') for row in rows]
            if not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
                raise ValueError('Every spec needs a unique output_name.')
            specs[root] = rows
        attempt(key, read_specs)

    def actions():
        import numpy as np
        if 'action_config' not in specs:
            return
        rows = specs['action_config']
        for row in rows:
            if not row.get('actuator_name'):
                raise ValueError('Each action spec needs actuator_name.')
            low, high = row.get('min_value', -1), row.get('max_value', 1)
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError('Action bounds must be finite and min_value < max_value.')
        details['actuators'] = len(rows)
        from library.thruster_health import health_settings
        enabled, health = health_settings(config.get('action', {}), len(rows))
        details['thruster_health_observations'] = len(rows) if enabled else 0
        if enabled:
            details['thruster_health'] = health
        if config.get('action', {}).get('force_6Dof', False):
            matrix = np.asarray(config['action']['tcm'], dtype=float)
            if matrix.ndim != 2 or not matrix.size or matrix.shape[1] != len(rows) or not np.isfinite(matrix).all():
                raise ValueError('TCM must have one column per actuator, with finite entries.')
            details['vehicle_actions'] = int(matrix.shape[0])
        else:
            details['vehicle_actions'] = len(rows)
        details['raw_state_values'] = len(specs.get('state_config', []))
    attempt('Actions', actions)

    def training():
        from library.training_seeding import training_seed
        training_seed(config)
        train, model = config['train'], config['model']
        number(train['total_timesteps'], 'library.training.total_timesteps', integer=True)
        algorithm = str(model['algorithm']).upper()
        if algorithm not in ('PPO', 'SAC', 'TD3'):
            raise ValueError('Training algorithm must be PPO, SAC or TD3.')
        workers = number(train.get('eval_instances', 1), 'library.training.eval_instances', integer=True)
        episodes = number(train.get('eval_episodes', 5), 'library.training.eval_episodes', integer=True)
        number(train['eval_freq'], 'library.training.eval_freq', integer=True)
        details.update(evaluation_workers=workers, evaluation_episodes=episodes, seed=train.get('seed'))
        if config['env']['base_port'] + config['env']['instances'] + workers >= 65536:
            raise ValueError('Training and evaluation port range exceeds 65535.')
        threshold = train.get('success_rate_threshold')
        if threshold is not None:
            number(threshold, 'library.training.success_rate_threshold', inclusive=True)
            if threshold >= 1:
                raise ValueError('Success threshold must be < 1 (the comparison is strictly greater).')
            details['successes_to_stop'] = math.floor(threshold * episodes) + 1
            required = train.get('success_rate_required_evaluations', 3)
            if type(required) is not int or required < 1:
                raise ValueError('train.success_rate_required_evaluations must be a positive integer.')
            floor = train.get('success_rate_drop_threshold', min(0.6, threshold))
            number(floor, 'train.success_rate_drop_threshold', inclusive=True)
            if floor > threshold:
                raise ValueError('Success drop threshold must not exceed the success threshold.')
            details.update(success_evaluations_required=required, success_drop_threshold=floor)
        if algorithm == 'PPO':
            steps = number(model['n_steps'], 'model.n_steps', integer=True)
            batch = number(model['batch_size'], 'model.batch_size', integer=True, minimum=2, inclusive=True)
            rollout = steps * config['env']['instances']
            if rollout < 2:
                raise ValueError('PPO rollout size must be at least two samples.')
            details['rollout_samples'] = rollout
            if rollout % batch:
                warnings.append('PPO rollout (%d) is not divisible by batch_size (%d); the final minibatch will be smaller.' % (rollout, batch))
        if config.get('log', {}).get('enable_wandb', False) and importlib.util.find_spec('wandb') is None:
            raise ValueError('W&B is enabled but not installed. Install wandb or disable log.enable_wandb.')
    if mode == 'training':
        attempt('Training', training)

    def checkpoint():
        cfg = config.get('model', {}) if mode == 'training' else config.get('evaluate', {})
        if mode == 'training' and not cfg.get('pretrained', False):
            return
        if mode == 'testing':
            return  # Random/manual testing does not need a policy.
        path = cfg.get('model_path')
        if str(cfg.get('algorithm', '')).upper() == 'PID' or str(path).upper() == 'PID':
            if not supports_fixed_pid(config):
                raise ValueError('The fixed PID baseline only supports docking.')
            return
        if not path:
            raise ValueError('Select a model_path for this workflow.')
        resolved = Path(resolve_model_path(path))
        if not resolved.is_file() and not Path(str(resolved) + '.zip').is_file():
            raise ValueError('Checkpoint not found: %s' % resolved)
        warnings.append('Checkpoint exists; observation/action compatibility is checked when the environment attaches.')
    attempt('Checkpoint', checkpoint)
    exe = Path(resolve_path('build/StonefishRL'))
    if not exe.is_file() or not os.access(str(exe), os.X_OK):
        errors.append('Simulator: build/StonefishRL is missing or not executable; build the C++ server first.')
    randomization = config.get('domain_randomization', {})
    if randomization.get('enabled'):
        for key in ('config', 'editor'):
            attempt('Domain randomization ' + key, lambda key=key: file_path('domain_randomization', key))
    return dict(ok=not errors, details=details, errors=errors, warnings=warnings)


