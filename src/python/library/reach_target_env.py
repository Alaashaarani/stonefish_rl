"""Small example task: bring a configured robot to a fixed world-frame target."""
import math
import numpy as np

from library.task_env import StonefishTaskEnv, TaskStep


class ReachTargetEnv(StonefishTaskEnv):
    def validate_task(self):
        cfg = self.task_config
        self.position_names = cfg.get('position_names', ['position_x', 'position_y', 'position_z'])
        if len(self.position_names) != 3 or any(n not in self.state_indices for n in self.position_names):
            raise ValueError('task.position_names must select three existing state output names (world x/y/z).')
        self.target = np.asarray(cfg['target_position'], dtype=np.float32)
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError('task.target_position must contain three finite world coordinates.')
        self.tolerance = float(cfg.get('tolerance', 0.2))
        self.hold_seconds = float(cfg.get('hold_seconds', 1.0))
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('task.tolerance must be positive and finite.')
        if not math.isfinite(self.hold_seconds) or self.hold_seconds < 0:
            raise ValueError('task.hold_seconds must be nonnegative and finite.')
        self.weights = self.config.get('reward', {}).get('weights', [1.0, 0.01, 10.0])
        if len(self.weights) != 3 or not np.isfinite(self.weights).all():
            raise ValueError('ReachTargetEnv reward.weights needs [distance, effort, success_bonus].')

    def reset_task(self, options):
        self.hold_steps = 0

    def evaluate_transition(self, previous_state, action):
        delta = self.state_values(self.position_names) - self.target
        distance = float(np.linalg.norm(delta))
        self.hold_steps = self.hold_steps + 1 if distance <= self.tolerance else 0
        success = bool(self.hold_steps >= max(1, math.ceil(self.hold_seconds * self.rl_observation_freq)))
        reward = -(self.weights[0] * distance + self.weights[1] * float(action @ action)) / self.rl_observation_freq
        if success:
            reward += self.weights[2]
        return TaskStep(reward, terminated=success, success=success,
                        info={'target_distance': distance, 'target_error': delta.tolist(),
                              'hold_seconds': self.hold_steps / self.rl_observation_freq})
