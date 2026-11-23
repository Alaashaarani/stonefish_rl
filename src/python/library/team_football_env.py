"""Single-robot ball pushing with the future ten-policy observation/action layout."""
import math
import gymnasium as gym
import numpy as np

from library.task_env import StonefishTaskEnv, TaskStep
from library.team_football import observation, MessageBus, OBSERVATION_SIZE, ACTION_SIZE

POSE_NAMES = ('robot_x', 'robot_y', 'robot_z', 'robot_roll', 'robot_pitch', 'robot_yaw')
VELOCITY_NAMES = ('velocity_x', 'velocity_y', 'velocity_z', 'angular_x', 'angular_y', 'angular_z')
BALL_NAMES = ('ball_x', 'ball_y', 'ball_z')


class TeamFootballSoloEnv(StonefishTaskEnv):
    def validate_task(self):
        cfg = self.task_config
        if cfg.get('mode', 'solo') != 'solo':
            raise ValueError('Only solo physics is implemented. Ten-agent matches need the shared match adapter and scene.')
        if self.history_length or self.observe_actions:
            raise ValueError('Team football uses a fixed 62-value observation; disable history and observe_actions.')
        if self.tcm is None or self.tcm.shape[0] != 4:
            raise ValueError('Team football requires a four-row TCM for fx, fy, fz, fyaw.')
        self.state_values(POSE_NAMES + VELOCITY_NAMES + BALL_NAMES)
        self.goal = np.asarray(cfg['goal_position'], dtype=np.float32)
        if self.goal.shape != (3,) or not np.isfinite(self.goal).all():
            raise ValueError('task.goal_position must contain three finite coordinates.')
        self.goal_tolerance = float(cfg.get('goal_tolerance', .5))
        if not math.isfinite(self.goal_tolerance) or self.goal_tolerance <= 0:
            raise ValueError('goal_tolerance must be positive and finite.')
        self.weights = np.asarray(self.config.get('reward', {}).get('weights', [1., 2., .01, 25.]), dtype=float)
        if self.weights.shape != (4,) or not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError('reward.weights: four nonnegative values [approach, ball_progress, effort, goal].')
        self.action_size = ACTION_SIZE
        self.action_space = gym.spaces.Box(-1., 1., (ACTION_SIZE,), dtype=np.float32)
        self.obs_size = self.total_obs_size = OBSERVATION_SIZE
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), dtype=np.float32)
        self.observation_names = list(POSE_NAMES + VELOCITY_NAMES + BALL_NAMES)
        self.observation_names += ['teammate_%d_%s' % (i, a) for i in range(1, 5) for a in 'xyz']
        self.observation_names += ['opponent_%d_%s' % (i, a) for i in range(1, 6) for a in 'xyz']
        self.observation_names += ['teammate_%d_message_%d' % (i, j) for i in range(1, 5) for j in range(1, 6)]
        self.policy_action_names = ['fx', 'fy', 'fz', 'fyaw'] + ['message_%d' % i for i in range(1, 6)]
        self.messages = MessageBus()
        self.current_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    def _process_action_vector(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Expected nine finite actions within [-1, 1].')
        # Communications must never be sent to a thruster. Commands are normalized,
        # not Newtons: retain the existing task TCM and respect actuator limits.
        commands = self.tcm_inv @ action[:4]
        specs = self.action_config['action_config']['specs']
        return np.clip(commands, [s.get('min_value', -1.) for s in specs],
                       [s.get('max_value', 1.) for s in specs])

    def reset_task(self, options):
        self.messages.reset()
        self.team_points = 0.

    def build_observation(self):
        self.observation = observation('blue_1', {'blue_1': self.state_values(POSE_NAMES)},
            {'blue_1': self.state_values(VELOCITY_NAMES)}, self.state_values(BALL_NAMES), self.messages.messages)
        return self.observation

    def evaluate_transition(self, previous_state, action):
        self.messages.commit({'blue_1': action})
        ball = self.state_values(BALL_NAMES)
        previous_ball = self.state_values(BALL_NAMES, previous_state)
        robot = self.state_values(POSE_NAMES)[:3]
        previous_robot = self.state_values(POSE_NAMES, previous_state)[:3]
        distance = float(np.linalg.norm(ball - self.goal))
        approach = float(np.linalg.norm(previous_robot-previous_ball) - np.linalg.norm(robot-ball))
        progress = float(np.linalg.norm(previous_ball-self.goal)) - distance
        effort = -float(action[:4] @ action[:4]) / self.rl_observation_freq
        success = bool(distance <= self.goal_tolerance)
        components = np.array([approach, progress, effort, float(success)]) * self.weights
        self.team_points = 25. if success else 0.
        return TaskStep(float(components.sum()), terminated=success, success=success, info={
            'team_points': self.team_points, 'ball_goal_distance': distance,
            'robot_ball_distance': float(np.linalg.norm(robot-ball)),
            'reward_components': dict(zip(('approach', 'ball_progress', 'effort', 'goal'), components.tolist())),
            'outgoing_message': action[4:].tolist(), 'game_mode': 'solo'})
