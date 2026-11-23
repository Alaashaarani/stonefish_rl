"""Fixed policy interface for solo pretraining and a future shared 5-vs-5 match.

World positions are metres, orientations radians; velocities use the simulator's
odometry convention. Stable roster slots are never sorted by distance.
"""
import numpy as np

OBSERVATION_SIZE = 62
ACTION_SIZE = 9
AGENTS = tuple('%s_%d' % (team, i) for team in ('blue', 'red') for i in range(1, 6))
ROLES = dict(zip(AGENTS, ('fast', 'fast', 'agile', 'agile', 'heavy') * 2))


def teammates(agent):
    if agent not in AGENTS:
        raise ValueError('Unknown agent: %s' % agent)
    return tuple(name for name in AGENTS if name != agent and name.split('_')[0] == agent.split('_')[0])


def vector(value, size, label):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('%s must contain %d finite values.' % (label, size))
    return result


def observation(agent, poses, velocities, ball, messages, visibility_radius=3.0):
    """Build [pose6, velocity6, ball3, teammates12, opponents15, messages20].

    Absent robots and opponents beyond the 3D radius have zero position slots.
    Radius-only visibility is used; occlusion and viewing angle are not modelled.
    Messages are the previous joint step's broadcasts (provided by MessageBus).
    """
    friends = teammates(agent)
    if not np.isfinite(visibility_radius) or visibility_radius <= 0:
        raise ValueError('visibility_radius must be positive and finite.')
    own = vector(poses[agent], 6, 'pose')
    values = [own, vector(velocities[agent], 6, 'velocity'), vector(ball, 3, 'ball')]
    values.extend(vector(poses[name], 6, 'teammate pose')[:3] if name in poses else np.zeros(3) for name in friends)
    for name in AGENTS:
        if name == agent or name in friends:
            continue
        position = vector(poses[name], 6, 'opponent pose')[:3] if name in poses else None
        values.append(position if position is not None and np.linalg.norm(position-own[:3]) <= visibility_radius else np.zeros(3))
    values.extend(vector(messages.get(name, np.zeros(5)), 5, 'message') if name in poses else np.zeros(5) for name in friends)
    return np.concatenate(values).astype(np.float32)


class MessageBus:
    """Commit all broadcasts together after a shared physics step, then observe."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.messages = {name: np.zeros(5, dtype=np.float32) for name in AGENTS}

    def commit(self, actions):
        if set(actions) - set(AGENTS):
            raise ValueError('Unknown agents in joint actions.')
        updated = {}
        for name in AGENTS:
            action = vector(actions[name], ACTION_SIZE, 'action') if name in actions else np.zeros(ACTION_SIZE)
            if np.any(np.abs(action) > 1):
                raise ValueError('Actions must lie within [-1, 1].')
            updated[name] = action[4:].copy()
        self.messages = updated
