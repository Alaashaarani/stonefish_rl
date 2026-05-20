"""Football task environment.

This file is intentionally small and direct so it can be used as a template for
new Stonefish RL tasks. When creating another task, the important functions to
edit are:

- `build_reset_command`: choose which simulated objects are respawned and where.
- `reset`: reset task counters and build the first observation.
- `step`: compute the task reward, termination, truncation, and info values.
- `_get_state_by_pattern`: change this only if your observation naming scheme
  cannot be matched with simple substrings.
"""

import time

import numpy as np

from stonefish_rl.envs.base_env import EnvStonefishRLParallel, launch_stonefish_simulator


class FootballEnv(EnvStonefishRLParallel):
    """A simple task where Girona pushes a ball toward a fixed goal.

    Reward is the negative weighted sum of:

    - Euclidean distance between the robot and the ball.
    - Euclidean distance between the ball and the goal.

    The goal is configured as a fixed position in YAML because the current C++
    state extraction supports robot-to-robot relative poses, while the provided
    goal scenario is a static object.
    """

    def __init__(self, rank, config, **kwargs):
        self.state_path = config["env"]["state_config"]
        self.act_path = config["env"]["action_config"]
        self.episode_duration = config["env"]["episode_duration"]
        self.rl_observation_freq = config["env"]["rl_observation_freq"]

        self.process = launch_stonefish_simulator(rank, config)
        time.sleep(1.0)

        if config["action"]["force_6Dof"]:
            self.tcm = np.array(config["action"]["tcm"], dtype=np.float32)
            action_size = self.tcm.shape[0]
        else:
            self.tcm = None
            action_size = None

        self.observe_actions = config["env"]["observe_actions"]
        self.history_length = max(0, int(config["env"]["history_length"]))

        super().__init__(action_size, env_id=rank, base_port=config["env"]["base_port"])

        task_config = config.get("football", {})
        self.goal_position = np.array(
            task_config.get("goal_position", [-5.5, 0.0, 5.2]),
            dtype=np.float32,
        )
        self.goal_tolerance = float(task_config.get("goal_tolerance", 0.5))
        self.distance_scale = float(task_config.get("distance_scale", 10.0))
        self.robot_ball_weight = float(task_config.get("robot_ball_weight", 1.0))
        self.ball_goal_weight = float(task_config.get("ball_goal_weight", 1.0))
        self.success_bonus = float(task_config.get("success_bonus", 50.0))
        self.randomize_reset = bool(task_config.get("randomize_reset", True))
        self.robot_start_position = np.array(
            task_config.get("robot_start_position", [0.0, 0.0, 0.7]),
            dtype=np.float32,
        )
        self.ball_start_position = np.array(
            task_config.get("ball_start_position", [-2.0, 0.0, 5.2]),
            dtype=np.float32,
        )
        self.robot_random_xy = np.array(
            task_config.get("robot_random_xy", [2.0, 2.0]),
            dtype=np.float32,
        )
        self.ball_random_xy = np.array(
            task_config.get("ball_random_xy", [1.0, 1.0]),
            dtype=np.float32,
        )

        self.enable_currents = bool(config["sim"].get("current", False))
        self.current_x = float(config["sim"].get("current_value", [0.0, 0.0])[0])
        self.current_y = float(config["sim"].get("current_value", [0.0, 0.0])[1])
        self.current_uniform = bool(config["sim"].get("current_uniform", False))

        self.step_count = 0
        self.info = {}
        self.observation_history = []
        self.current_action = np.zeros(self.action_size, dtype=np.float32)

    def update_runtime_params(self, params):
        """Apply optional GUI/test overrides without changing YAML behavior."""
        super().update_runtime_params(params)

        if "reward_weights" in params:
            weights = list(params["reward_weights"])
            if len(weights) >= 1:
                self.robot_ball_weight = float(weights[0])
            if len(weights) >= 2:
                self.ball_goal_weight = float(weights[1])
        if "robot_ball_weight" in params:
            self.robot_ball_weight = float(params["robot_ball_weight"])
        if "ball_goal_weight" in params:
            self.ball_goal_weight = float(params["ball_goal_weight"])
        if "goal_position" in params:
            self.goal_position = np.array(params["goal_position"], dtype=np.float32)
        if "goal_tolerance" in params:
            self.goal_tolerance = float(params["goal_tolerance"])
        if "current_enabled" in params:
            self.enable_currents = bool(params["current_enabled"])
        if "current_value" in params:
            value = params["current_value"]
            self.current_x = float(value[0])
            self.current_y = float(value[1])
        if "current_uniform" in params:
            self.current_uniform = bool(params["current_uniform"])
        if "randomize_reset" in params:
            self.randomize_reset = bool(params["randomize_reset"])
        if "robot_start_position" in params:
            self.robot_start_position = np.array(params["robot_start_position"], dtype=np.float32)
        if "target_start_position" in params:
            self.ball_start_position = np.array(params["target_start_position"], dtype=np.float32)

        return self.runtime_params

    def build_reset_command(self):
        """Return Stonefish reset commands for the robot and ball."""
        robot_pos = self.robot_start_position.copy()
        ball_pos = self.ball_start_position.copy()

        if self.randomize_reset:
            robot_pos[:2] += self.np_random.uniform(-self.robot_random_xy, self.robot_random_xy)
            ball_pos[:2] += self.np_random.uniform(-self.ball_random_xy, self.ball_random_xy)

        if self.enable_currents:
            if self.current_uniform:
                current_vec = [
                    self.np_random.uniform(-self.current_x, self.current_x),
                    self.np_random.uniform(-self.current_y, self.current_y),
                    0.0,
                ]
            else:
                current_vec = [self.current_x, self.current_y, 0.0]
        else:
            current_vec = [0.0, 0.0, 0.0]

        return [
            {
                "name": "girona1000",
                "position": robot_pos.tolist(),
                "rotation": [0.0, 0.0, 0.0],
                "current": current_vec,
            },
            {
                "name": "ball",
                "position": ball_pos.tolist(),
                "rotation": [0.0, 0.0, 0.0],
            },
        ]

    def reset(self, seed=None, options=None):
        """Reset task state and return the first normalized observation."""
        super().reset(seed=seed)
        self.step_count = 0
        self.info = {}
        self.observation_history = []
        self.current_action = np.zeros(self.action_size, dtype=np.float32)

        obs = self._build_observation(self.state)
        return obs, self.info

    def step(self, action):
        """Advance Stonefish, calculate football reward, and return Gym output."""
        self.step_count += 1
        self.current_action = np.array(action, dtype=np.float32).flatten()

        raw_obs, _, _, _ = super().step(action)
        robot_ball_vector = self._get_state_by_pattern("robot_to_ball", raw_obs)
        ball_position = self._get_state_by_pattern("ball_position", raw_obs)

        robot_ball_distance = float(np.linalg.norm(robot_ball_vector[:3]))
        ball_goal_distance = float(np.linalg.norm(ball_position[:3] - self.goal_position))

        reward = -(
            self.robot_ball_weight * robot_ball_distance
            + self.ball_goal_weight * ball_goal_distance
        ) / self.rl_observation_freq

        terminated = ball_goal_distance <= self.goal_tolerance
        if terminated:
            reward += self.success_bonus

        truncated = self.step_count / self.rl_observation_freq >= self.episode_duration

        self.info = {
            "robot_ball_distance": robot_ball_distance,
            "ball_goal_distance": ball_goal_distance,
            "goal_position": self.goal_position.tolist(),
            "step": self.step_count,
        }

        obs = self._build_observation(raw_obs)
        return obs, reward, terminated, truncated, self.info

    def _build_observation(self, raw_obs):
        obs = np.array(raw_obs, dtype=np.float32).copy()
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.distance_scale, neginf=-self.distance_scale)

        if self.observe_actions:
            obs = np.concatenate((obs, self.current_action))

        obs = np.clip(obs, -self.distance_scale, self.distance_scale) / self.distance_scale
        return self._concatenate_history(obs)

    def _concatenate_history(self, obs):
        self.observation_history.append(obs)
        if len(self.observation_history) > self.history_length + 1:
            self.observation_history.pop(0)

        observation = np.concatenate(self.observation_history, axis=0)
        if len(observation) < self.total_obs_size:
            padding = np.zeros(self.total_obs_size - len(observation), dtype=np.float32)
            observation = np.concatenate((observation, padding))

        self.observation = observation.astype(np.float32)
        return self.observation

    def _get_state_by_pattern(self, pattern, state=None):
        values = []
        source_state = self.state if state is None else state

        for index, name in enumerate(self.state_names):
            if pattern in name and index < len(source_state):
                values.append(source_state[index])

        return np.array(values, dtype=np.float32) if values else np.zeros(3, dtype=np.float32)
