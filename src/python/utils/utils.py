import pygame
import numpy as np
import matplotlib.pyplot as plt
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
        fwd = self.joystick.get_axis(self.axes["fwd"])
        lat = self.joystick.get_axis(self.axes["lat"])
        ver = -self.joystick.get_axis(self.axes["ver"])
        pitch = self.joystick.get_axis(self.axes["pitch1"])-self.joystick.get_axis(self.axes["pitch2"])
        yaw = -self.joystick.get_axis(self.axes["yaw"])
        
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


class RealTimePlotter:
    def __init__(self, num_curves=3, max_entries=100):
        self.max_entries = max_entries
        self.num_curves = num_curves
        
        # Initialize empty lists for X and Y data
        self.x_data = []
        self.y_data = [[] for _ in range(num_curves)]
        
        # Setup the plot in interactive mode
        plt.ion() 
        self.fig, self.ax = plt.subplots()
        
        # Create line objects (initially empty)
        self.lines = []
        for i in range(num_curves):
            ln, = self.ax.plot([], [], label=f"Curve {i+1}")
            self.lines.append(ln)
            
        self.ax.set_xlim(0, max_entries+50)
        self.ax.set_ylim(-1.0, 1.0) # Set your expected range (or use autoscale)
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        
        self.counter = 0

    def update(self, new_values):
        """
        Call this inside your loop. 
        new_values: a list/array of length num_curves
        """
        self.counter += 1
        
        # 1. Append new data
        self.x_data.append(self.counter)
        for i in range(self.num_curves):
            self.y_data[i].append(new_values[i])
            
        # 2. Dump old values if we exceed max_entries
        if len(self.x_data) > self.max_entries:
            self.x_data.pop(0)
            for i in range(self.num_curves):
                self.y_data[i].pop(0)
        
        # 3. Update the line data pointers (very fast)
        for i, line in enumerate(self.lines):
            line.set_data(self.x_data, self.y_data[i])

        # 4. Slide the X-axis window
        if self.counter > self.max_entries:
            self.ax.set_xlim(self.x_data[0], self.x_data[-1]+50)
            
        # 5. Refresh the visual
        # Adjust Y-axis if values go out of bounds
        current_min = min(min(y) for y in self.y_data)
        current_max = max(max(y) for y in self.y_data)
        if current_min != self.ax.get_ylim()[0] or current_max != self.ax.get_ylim()[1]:
            self.ax.set_ylim(current_min - 1, current_max + 1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def global_path(relative_path):
    """Get absolute path from project root"""
    # Path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

    return os.path.join(project_root, relative_path)