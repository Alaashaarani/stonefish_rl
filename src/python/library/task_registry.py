"""Task capabilities used by launchers without importing or starting environments."""
import math

DOCKING_TYPES = frozenset(('docking', 'ds', 'docking_ds'))
DOCKING_FAMILY = DOCKING_TYPES | {'pid_tuning', 'pid_rl_blend'}
REWARD_NAMES = {
    **{name: ('xy', 'z', 'yaw', 'docking_accuracy', 'smoothness', 'velocity', 'collision')
       for name in DOCKING_FAMILY},
    'football': ('robot_ball_distance', 'ball_goal_distance'),
    'reach_target': ('distance', 'effort', 'success_bonus'),
    'team_football_solo': ('approach', 'ball_progress', 'effort', 'goal'),
}


def task_type(config):
    return str(config.get('env', {}).get('type', 'docking')).strip().lower()


def uses_docking_observations(config):
    return not config.get('env', {}).get('entry_point') and task_type(config) in DOCKING_FAMILY


def supports_fixed_pid(config):
    return not config.get('env', {}).get('entry_point') and task_type(config) in DOCKING_TYPES


def reward_names(config):
    names = config.get('env', {}).get('reward_names')
    if names is not None:
        if (not isinstance(names, list) or not names or
                not all(isinstance(name, str) and name.strip() for name in names) or
                len(set(names)) != len(names)):
            raise ValueError('env.reward_names must be a nonempty list of unique names.')
        return tuple(names)
    if config.get('env', {}).get('entry_point'):
        return None
    return REWARD_NAMES.get(task_type(config))


def is_reward_vector(value):
    return (isinstance(value, list) and bool(value) and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in value))


def validate_task_reward(config, context='Task'):
    names = reward_names(config)
    weights = (config.get('reward') or {}).get('weights')
    if weights is None:
        return
    if not is_reward_vector(weights) or (names is not None and len(weights) != len(names)):
        expected = 'a nonempty list of finite numbers' if names is None else '%d finite numbers (%s)' % (
            len(names), ', '.join(names))
        raise ValueError('%s reward.weights must contain %s. Got: %r' % (context, expected, weights))


def episode_success(info, terminated, *, legacy_termination=False):
    """Success is task-defined; termination can also mean failure."""
    if 'is_success' in info:
        value = info['is_success']
        # Accept NumPy scalar booleans without requiring NumPy in this module.
        if isinstance(value, bool) or type(value).__name__ == 'bool_':
            return bool(value)
        raise ValueError("info['is_success'] must be a boolean.")
    if legacy_termination:
        return bool(terminated)
    raise ValueError("Task must return info['is_success'] at the end of every episode.")
