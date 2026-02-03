import pygame
import numpy as np

class LogitechController:
    def __init__(self, deadzone=0.1):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No Logitech controller found! Is it plugged in? or disable it in the config.")
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.deadzone = deadzone
        
        # Mapping for Logitech F310/F710 (Mode light OFF)
        # Standard: 0:LX, 1:LY, 4:RX, 3:RY
        self.axes = {
            "fwd": 1,  # Left Stick Y
            "lat": 0,  # Left Stick X
            "ver": 3,  # Right Stick Y
            "yaw": 4   # Right Stick X
        }
        print(f"Controller Initialized: {self.joystick.get_name()}")

    def get_thruster_values(self):
        """
        Returns a 5-element numpy array: [Forward, Lateral, Vertical, Yaw, Pitch]
        Values are normalized between -1.0 and 1.0.
        """
        pygame.event.pump() # Necessary to update internal axis states
        
        # Read axes (Note: Pygame returns 1.0 for 'Down', so we invert Y axes)
        fwd = self.joystick.get_axis(self.axes["fwd"])
        lat = -self.joystick.get_axis(self.axes["lat"])
        ver = self.joystick.get_axis(self.axes["ver"])
        yaw = -self.joystick.get_axis(self.axes["yaw"])
        T1 = np.clip(fwd + lat/2,-1,1)
        T2 = np.clip(fwd - lat/2,-1,1)
        T3 = ver
        T4 = yaw
        T5 = yaw
        # Create action vector [Forward, Lateral, Vertical, Yaw, Pitch]
        # We keep Pitch at 0.0 unless you want to map it to triggers
        action = np.array([T1,T2,T3,T4,T5], dtype=np.float32)
        
        # Apply deadzone to stop 'drifting' when sticks are released
        action[np.abs(action) < self.deadzone] = 0.0
        
        return action
    
if __name__ == "__main__":
    controller = LogitechController(deadzone=0.1)
    print("Starting to read controller inputs. Press Ctrl+C to exit.")
    
    try:
        while True:
            thruster_values = controller.get_thruster_values()
            print(f"Thruster Values: {np.round(thruster_values,2)}", end='\r')
            pygame.time.wait(100)  # Polling delay
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        pygame.quit()