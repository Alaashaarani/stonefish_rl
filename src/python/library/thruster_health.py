"""Validate and sample per-episode actuator effectiveness fractions."""
import math
import numpy as np


def health_settings(action, actuator_count):
    enabled = action.get('use_thruster_health', False)
    if type(enabled) is not bool:
        raise ValueError('action.use_thruster_health must be true or false.')
    settings = action.get('thruster_health', {})
    if not isinstance(settings, dict):
        raise ValueError('action.thruster_health must be a mapping.')
    unknown = set(settings) - {'min', 'max', 'max_affected'}
    if unknown:
        raise ValueError('Unknown action.thruster_health settings: %s' % sorted(unknown))
    low, high = settings.get('min', .2), settings.get('max', 1.)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (low, high)) or not 0 <= low <= high <= 1:
        raise ValueError('action.thruster_health requires 0 <= min <= max <= 1.')
    maximum = settings.get('max_affected', min(2, actuator_count))
    if type(maximum) is not int or not 0 <= maximum <= actuator_count:
        raise ValueError('action.thruster_health.max_affected must be an integer between 0 and %d.' % actuator_count)
    return enabled, dict(min=float(low), max=float(high), max_affected=maximum)


def sample_health(rng, actuator_count, enabled, settings):
    health = np.ones(actuator_count, dtype=np.float32)
    if enabled and settings['max_affected']:
        count = int(rng.integers(1, settings['max_affected'] + 1))
        indices = rng.choice(actuator_count, size=count, replace=False)
        health[indices] = rng.uniform(settings['min'], settings['max'], count).astype(np.float32)
    return health
