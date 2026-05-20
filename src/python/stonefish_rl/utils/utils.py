import pygame
import numpy as np
import os

class LogitechController:
    def __init__(self, use_forces, deadzone=0.1):
        pygame.init()
        pygame.joystick.init()
        
        self.forces_6dof = use_forces 

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No Logitech controller found! Is it plugged in? or disable it in the config.")
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.deadzone = deadzone
        
        # Mapping for Logitech F310/F710 (Mode light OFF)
        # Standard: 0:LX, 1:LY, 4:RY, 3:RX
        self.axes = {
            "fwd": 1,  # Left Stick Y
            "lat": 0,  # Left Stick X
            "ver": 4,  # Right Stick Y
            "pitch1": 2, # left trigger 
            "pitch2": 5, # left trigger 
            "yaw": 3   # Right Stick X
        }
        print(f"Controller Initialized: {self.joystick.get_name()}")

    def get_thruster_values(self):
        """
        Returns a 5-element numpy array: [Forward, Lateral, Vertical, Yaw, Pitch]
        Values are normalized between -1.0 and 1.0.
        """
        pygame.event.pump() # Necessary to update internal axis states
        
        # Read axes (Note: Pygame returns 1.0 for 'Down', so we invert Y axes)
        fwd = -self.joystick.get_axis(self.axes["fwd"])
        lat = self.joystick.get_axis(self.axes["lat"])
        ver = self.joystick.get_axis(self.axes["ver"])
        pitch = self.joystick.get_axis(self.axes["pitch1"])-self.joystick.get_axis(self.axes["pitch2"])
        pitch = np.clip(pitch, -1.0, 1.0)  # Ensure pitch is within bounds
        yaw = self.joystick.get_axis(self.axes["yaw"])
        
        if self.forces_6dof: 
            action = np.array([fwd,lat,ver,0.0,pitch,yaw], dtype=np.float32)
            action[np.abs(action) < self.deadzone] = 0.0
            return action
        else:
            T1 = np.clip(fwd + yaw/2,-1,1)
            T2 = np.clip(fwd - yaw/2,-1,1)
            T3 = lat
            T4 = np.clip(ver + pitch,-1,1)
            T5 = np.clip(ver - pitch,-1,1)
            # Create action vector [Forward, Lateral, Vertical, Yaw, Pitch]
            # We keep Pitch at 0.0 unless you want to map it to triggers
            action = np.array([T1,T2,T3,T4,T5], dtype=np.float32)
            
            # Apply deadzone to stop 'drifting' when sticks are released
            action[np.abs(action) < self.deadzone] = 0.0
            
            return action 

def project_root():
    """Return the stonefish_rl repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))


def resolve_path(path_value, base_dir=None):
    """Resolve absolute paths, repo-relative paths, and optional base-relative paths."""
    if path_value in (None, ""):
        return path_value

    path = os.path.expanduser(str(path_value))
    if os.path.isabs(path):
        return path

    if base_dir is not None:
        return os.path.abspath(os.path.join(base_dir, path))

    return os.path.join(project_root(), path)


def resolve_model_path(path_value):
    """Resolve model paths from the new models folder, repo root, or legacy src/python."""
    if path_value in (None, ""):
        return path_value

    path = os.path.expanduser(str(path_value))
    if os.path.isabs(path):
        return path

    candidates = [
        resolve_path(os.path.join("models", path)),
        resolve_path(path),
        resolve_path(os.path.join("src/python", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) or os.path.exists(candidate + ".zip"):
            return candidate

    return candidates[0]
