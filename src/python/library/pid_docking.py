"""PID baseline supporting AUV-frame errors or station-frame pose observations."""

import numpy as np
from library.pose_frames import pose_to_transform


def is_pid(algorithm, model_path):
    return str(algorithm).strip().upper() == "PID" or str(model_path).strip().upper() == "PID"


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DockingPID:
    """Four independent axes ordered x, y, z, yaw; outputs are normalized commands."""

    def __init__(self, dt, config=None):
        config = config or {}
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("PID timestep must be positive and finite")
        self.mode = config.get("mode", "one_stage")
        if self.mode not in ("one_stage", "two_stage"):
            raise ValueError("pid.mode must be one_stage or two_stage")
        self.threshold = float(config.get("alignment_threshold", 1.0))
        self.z_offset = float(config.get("z_offset", 0.5))
        self.offsets = np.array([0.0, 0.0, self.z_offset, 0.0])
        if not np.isfinite(self.z_offset):
            raise ValueError("pid.z_offset must be finite")
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("pid.alignment_threshold must be positive and finite")
        for key, default in (
            ("kp", [0.4, 0.4, 0.5, 0.5]),
            ("ki", [0.02, 0.02, 0.02, 0.02]),
            ("kd", [0.15, 0.15, 0.15, 0.15]),
            ("output_limit", [1.0] * 4),
            ("integral_limit", [1.0] * 4),
        ):
            value = np.asarray(config.get(key, default), dtype=float)
            if value.shape != (4,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"pid.{key} must contain four finite nonnegative values: x, y, z, yaw")
            setattr(self, key, value)
        self.reset()

    def reset(self):
        # Start each episode with no accumulated error or derivative history.
        self.integral = np.zeros(4)
        self.previous = None
        self.active = np.zeros(4, dtype=bool)
        # One-stage enables all axes; two-stage starts with x/y/yaw.
        self.aligned = self.mode == "one_stage"

    def update(self, error):
        """Compute u = Kp * e + Ki * integral(e dt) + Kd * de/dt per axis.

        Arrays are ordered [x, y, z, yaw], with errors in metres/radians and
        dt in seconds. P/I use offset-adjusted errors; D uses measured errors
        so changing a learned target offset does not cause a derivative spike.
        The final output is a bounded command, not a calibrated force.
        """
        error = np.asarray(error, dtype=float).copy()
        if error.shape != (4,) or not np.all(np.isfinite(error)):
            raise ValueError("PID requires four finite errors: x, y, z, yaw")
        error[3] = wrap(error[3])
        measured_error = error.copy()
        # Offsets shift the controller target only, not the environment's goal.
        error += self.offsets
        error[3] = wrap(error[3])
        # Switch on measured horizontal distance, before controller offsets.
        # This switch is latched: once aligned, all axes stay enabled until reset,
        # even if horizontal error later grows above the threshold.
        if np.linalg.norm(measured_error[:2]) < self.threshold:
            self.aligned = True
        # Stage 1: [True, True, False, True] -> x/y/yaw (z waits).
        # Stage 2 (or one-stage mode): all True -> x/y/z/yaw together.
        # x/y are never disabled; they keep correcting while z/yaw align.
        active = np.array([True, True, self.aligned, True])
        derivative = np.zeros(4)
        if self.previous is not None:
            # Offset changes move the target, but must not create derivative kicks.
            difference = measured_error - self.previous
            # Use the shortest angular difference across the -pi/pi boundary.
            difference[3] = wrap(difference[3])
            derivative = difference / self.dt
        # self.active is the PREVIOUS step's mask: suppress D on the first sample
        # and on the step that enables z. Its P/I terms can act immediately.
        derivative[~self.active] = 0
        # Integrate e * dt, bounded per axis; inactive axes must not accumulate
        # error while waiting for stage 2.
        candidate = np.clip(self.integral + error * self.dt,
                            -self.integral_limit, self.integral_limit)
        candidate[~active] = 0
        output = self.kp * error + self.ki * candidate + self.kd * derivative
        # Anti-windup: reject this step's integral update if the proposed command
        # exceeds its limit and the error would push it further into saturation.
        # Opposite-sign error may still integrate to unwind the stored integral.
        winding_up = (np.abs(output) > self.output_limit) & (error * output > 0)
        self.integral = np.where(winding_up, self.integral, candidate)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.output_limit, self.output_limit)
        # This mask removes ALL P/I/D contributions from inactive axes.
        # Zero z command does not actively hold depth in stage 1.
        output[~active] = 0
        self.previous = measured_error
        self.active = active
        return output.astype(np.float32)


class PIDDockingPolicy:
    """SB3-style adapter; keeps observation noise, history, and TCM handling intact."""

    def __init__(self, env, config=None):
        if env.tcm is None or env.action_size not in (4, 6):
            raise ValueError("PID requires force_6Dof with a 4-row (x,y,z,yaw) or 6-row TCM")
        self.env = env
        self.frame = getattr(env, "observation_frame", "auv")
        # Desired AUV pose in the station frame; equivalent to the legacy docking offsets.
        self.target_pose = np.asarray((config or {}).get(
            "target_pose", [0, 0, -0.91, 0, 0, np.pi / 2]), dtype=float)
        pose_to_transform(self.target_pose)  # Validate before starting evaluation.
        if self.frame == "docking_station" and not {"relative_roll", "relative_pitch"}.issubset(env.policy_state_names):
            raise ValueError("Station-frame PID requires relative_roll and relative_pitch observations")
        self.controller = DockingPID(1.0 / env.rl_observation_freq, config)

    def reset(self):
        self.controller.reset()

    def predict(self, observation, deterministic=True):
        env = self.env
        # dsEnv returns zero-filled history at reset, not a measured pose.
        if env.step_count == 0:
            return np.zeros(env.action_size, dtype=np.float32), None
        observation = np.asarray(observation)
        if getattr(env, "use_thruster_health", False):
            observation = observation[:-len(env.thruster_health)]
        frames = observation.reshape(-1, env.obs_size)
        latest = frames[0] if env.history_gap > 0 else frames[-1]
        state = dict(zip(env.policy_state_names, latest))
        xyz = [state[f"error_{axis}"] * 5.0 for axis in ("x", "y", "z")]
        yaw = (np.arctan2(state["error_yaw_sin"], state["error_yaw_cos"])
               if env.yaw_observation == "cos_sin" else state["error_yaw"] * np.pi)
        if self.frame == "docking_station":
            # Observations hold AUV pose, so control error is target minus pose.
            error = np.r_[self.target_pose[:3] - xyz, wrap(self.target_pose[5] - yaw)]
            command = self.controller.update(error)
            rotation = pose_to_transform([0, 0, 0, state["relative_roll"], state["relative_pitch"], yaw])[:3, :3]
            body_linear = rotation.T @ command[:3]
            body_angular = rotation.T @ np.array([0, 0, command[3]])
            action = np.r_[body_linear, body_angular] if env.action_size == 6 else np.r_[body_linear, body_angular[2]]
            # A four-axis TCM cannot apply the roll/pitch parts of station yaw torque.
            return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32), None
        command = self.controller.update(xyz + [yaw])
        action = np.zeros(env.action_size, dtype=np.float32)
        # Four-action TCM: [x,y,z,yaw]. Six-action TCM: [x,y,z,roll,pitch,yaw],
        # keeping roll/pitch zero. The environment maps this to thruster setpoints.
        action[[0, 1, 2, 3 if env.action_size == 4 else 5]] = command
        return np.clip(action, env.action_space.low, env.action_space.high), None
