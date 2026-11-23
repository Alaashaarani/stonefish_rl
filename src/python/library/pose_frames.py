"""Rigid transforms: poses use [x, y, z, roll, pitch, yaw] in metres/radians."""

import numpy as np
from scipy.spatial.transform import Rotation


def observation_frame(env_config):
    """The explicit Inverse switch takes precedence over the legacy frame name."""
    if "Inverse" in env_config:
        inverse = env_config["Inverse"]
        if not isinstance(inverse, bool):
            raise ValueError("env.Inverse must be a YAML boolean: true or false")
        return "docking_station" if inverse else "auv"
    frame = env_config.get("observation_frame", "auv")
    if frame not in ("auv", "docking_station"):
        raise ValueError("env.observation_frame must be auv or docking_station")
    return frame


def pose_to_transform(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("Pose must contain six finite values: x,y,z,roll,pitch,yaw")
    transform = np.eye(4)
    # Fixed-axis xyz equals Rz(yaw) @ Ry(pitch) @ Rx(roll), as in Stonefish.
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def invert_pose(pose):
    """Return the inverse pose and transform: T_ds_auv = inverse(T_auv_ds)."""
    transform = pose_to_transform(pose)
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    angles = Rotation.from_matrix(inverse[:3, :3]).as_euler("xyz")
    return np.concatenate((inverse[:3, 3], angles)), inverse
