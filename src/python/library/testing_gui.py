"""Interactive GUI for manually testing Stonefish RL environments."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from library.env_factory import make_env_instance, reload_docking_environment, ENV_CLASSES
from library.task_registry import supports_fixed_pid, uses_docking_observations, task_type
from library.common import LogitechController, resolve_path
from library.pid_docking import PIDDockingPolicy
from library.gui_panel import GuiPanel
from library.configuration import load_config as load_workflow_config, DEFAULT_CONFIG


def _parse_yaml_value(text, default):
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return default
    return default if value is None else value


def _as_float_list(text, size=None):
    value = _parse_yaml_value(text, [])
    if not isinstance(value, (list, tuple)):
        value = str(text).replace(",", " ").split()
    out = [float(v) for v in value]
    if size is not None and len(out) != size:
        raise ValueError(f"Expected {size} values, got {len(out)}")
    return out


class TestingGui(GuiPanel):
    """Small live-control GUI for stepping and plotting an environment."""

    def __init__(self, master=None):
        super().__init__(master, title="Stonefish RL Test GUI", geometry="1220x780")

        self.env = None
        self.starting = False
        self.config = None
        self.obs = None
        self.info = {}
        self.last_action = None
        self.running = False
        self.step_index = 0
        self.total_reward = 0.0
        self.plot_curves = []
        self.controller = None
        self.controller_error_shown = False
        self.loop_after_id = None
        self.policy = None
        self.policy_key = None

        self.env_type = tk.StringVar(value="config")
        self.config_path = tk.StringVar(value=DEFAULT_CONFIG)
        self.action_mode = tk.StringVar(value="zero")
        self.plot_key = tk.StringVar(value="reward")
        self.status = tk.StringVar(value="Idle")
        self.model_path = tk.StringVar(value="")
        self.algorithm = tk.StringVar(value="PPO")
        self.graphical = tk.BooleanVar(value=True)

        self.reward_weights = tk.StringVar(value="[1, 1, 1, 1, 1, 1, 1]")
        self.current_enabled = tk.BooleanVar(value=False)
        self.current_uniform = tk.BooleanVar(value=True)
        self.current_value = tk.StringVar(value="[0.0, 0.0]")
        self.offset_noise_enabled = tk.BooleanVar(value=False)
        self.random_noise_enabled = tk.BooleanVar(value=False)
        self.randomize_reset = tk.BooleanVar(value=True)
        self.robot_position = tk.StringVar(value="[0.0, 0.0, 0.7]")
        self.target_position = tk.StringVar(value="[0.0, 0.0, 5.2]")
        self.robot_rotation = tk.StringVar(value="[0.0, 0.0, 0.0]")
        self.target_rotation = tk.StringVar(value="[0.0, 0.0, 0.0]")

        self.action_sliders = []
        self._build_layout()

    def _build_layout(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        left_container = ttk.Frame(root)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_canvas = tk.Canvas(left_container, width=470, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_canvas.configure(yscrollcommand=scrollbar.set)
        left = ttk.Frame(left_canvas)
        window = left_canvas.create_window((0, 0), window=left, anchor=tk.NW)
        left.bind("<Configure>", lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(window, width=event.width))

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_env_panel(left)
        self._build_parameter_panel(left)
        self._build_action_panel(left)
        self._build_plot_panel(right)

        ttk.Label(root, textvariable=self.status, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _build_env_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Environment", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(frame, text="Type").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(
            frame,
            textvariable=self.env_type,
            values=["config"] + sorted(ENV_CLASSES),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky=tk.EW)

        ttk.Label(frame, text="Config").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.config_path, width=42).grid(
            row=1, column=1, sticky=tk.EW
        )
        ttk.Button(frame, text="Browse", command=self._browse_config).grid(row=1, column=2)
        ttk.Button(frame, text="Load YAML", command=self.load_yaml).grid(row=2, column=2)
        ttk.Checkbutton(frame, text="Simulator graphics", variable=self.graphical).grid(row=2, column=1, sticky=tk.W)

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        for index, (label, command) in enumerate((
            ("Start", self.start_env), ("Reset", self.reset_env), ("Step", self.step_once),
            ("Resume", self.resume), ("Pause", self.pause), ("Stop", self.stop_env),
        )):
            ttk.Button(buttons, text=label, command=command).grid(
                row=index // 3, column=index % 3, sticky=tk.EW, padx=2, pady=2
            )
        for column in range(3):
            buttons.columnconfigure(column, weight=1, uniform="environment_controls")

        frame.columnconfigure(1, weight=1)

    def _build_parameter_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Runtime Parameters", padding=8)
        self.runtime_parameter_frame = frame
        frame.pack(fill=tk.X, pady=(0, 8))

        rows = [
            ("Reward weights", self.reward_weights),
            ("Current [x, y]", self.current_value),
            ("Robot position", self.robot_position),
            ("Target/ball position", self.target_position),
            ("Robot rotation", self.robot_rotation),
            ("Target rotation", self.target_rotation),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W)
            ttk.Entry(frame, textvariable=var, width=32).grid(row=row, column=1, sticky=tk.EW)

        ttk.Checkbutton(frame, text="Enable current", variable=self.current_enabled).grid(
            row=len(rows), column=0, sticky=tk.W
        )
        ttk.Checkbutton(frame, text="Uniform random current", variable=self.current_uniform).grid(
            row=len(rows), column=1, sticky=tk.W
        )
        ttk.Checkbutton(frame, text="Offset noise", variable=self.offset_noise_enabled).grid(
            row=len(rows) + 1, column=0, sticky=tk.W
        )
        ttk.Checkbutton(frame, text="Random noise", variable=self.random_noise_enabled).grid(
            row=len(rows) + 1, column=1, sticky=tk.W
        )
        ttk.Checkbutton(frame, text="Randomize reset", variable=self.randomize_reset).grid(
            row=len(rows) + 2, column=0, sticky=tk.W
        )
        ttk.Button(frame, text="Apply Parameters", command=self.apply_params).grid(
            row=len(rows) + 2, column=1, sticky=tk.E
        )
        frame.columnconfigure(1, weight=1)

    def _build_action_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Action", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Combobox(
            frame,
            textvariable=self.action_mode,
            values=["zero", "random", "manual", "controller", "fixed PID", "trained model"],
            state="readonly",
            width=14,
        ).pack(anchor=tk.W)

        model = ttk.Frame(frame)
        model.pack(fill=tk.X, pady=4)
        ttk.Combobox(model, textvariable=self.algorithm, values=["PPO", "SAC", "TD3", "ONNX"],
                     state="readonly", width=7).pack(side=tk.LEFT)
        ttk.Entry(model, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model, text="Browse", command=self._browse_model).pack(side=tk.LEFT)
        self.action_slider_frame = ttk.Frame(frame)
        self.action_slider_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_plot_panel(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top, text="Plot").pack(side=tk.LEFT)
        self.plot_combo = ttk.Combobox(
            top,
            textvariable=self.plot_key,
            values=["reward", "total_reward"],
            width=30,
        )
        self.plot_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="New", command=self.new_plot).pack(side=tk.LEFT)
        ttk.Button(top, text="Add", command=self.add_plot_curve).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Remove", command=self.remove_plot_curve).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear Plot", command=self.clear_plot).pack(side=tk.LEFT)

        self.fig = Figure(figsize=(7.5, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Value")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        curve_frame = ttk.Frame(parent)
        curve_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(curve_frame, text="Curves").pack(side=tk.LEFT)
        self.curve_listbox = tk.Listbox(curve_frame, height=4, exportselection=False)
        self.curve_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        self.info_text = tk.Text(parent, height=10)
        self.info_text.pack(fill=tk.X, pady=(8, 0))

    def _browse_config(self):
        path = filedialog.askopenfilename(
            initialdir=resolve_path("include/parameters"),
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.config_path.set(path)

    def _load_config(self):
        config = load_workflow_config(self.config_path.get(), "testing")
        self.config = config
        self._load_defaults_from_config(config)
        if not self.model_path.get().strip():
            self._model_defaults(config)
        return config

    def _model_defaults(self, config):
        model = config.get("evaluate", {})
        self.model_path.set(model.get("model_path", ""))
        algorithm = str(model.get("algorithm", "PPO")).upper()
        self.algorithm.set(algorithm if algorithm in ("PPO", "SAC", "TD3", "ONNX") else "PPO")

    def load_yaml(self):
        try:
            config = self._load_config()
            self._model_defaults(config)
            self.status.set("YAML loaded; restart the environment to use changed settings")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _browse_model(self):
        path = filedialog.askopenfilename(initialdir=resolve_path("models"),
                                         filetypes=[("Models", "*.zip *.onnx")])
        if path:
            self.model_path.set(path)
            if path.lower().endswith(".onnx"):
                self.algorithm.set("ONNX")

    def _load_defaults_from_config(self, config):
        self._runtime_controls_supported = (uses_docking_observations(config) or
                                           (task_type(config) == 'football' and not config['env'].get('entry_point')))
        for widget in self.runtime_parameter_frame.winfo_children():
            if isinstance(widget, (ttk.Entry, ttk.Checkbutton, ttk.Button)):
                widget.configure(state=tk.NORMAL if self._runtime_controls_supported else tk.DISABLED)
        self.runtime_parameter_frame.configure(text=('Runtime Parameters' if self._runtime_controls_supported
                                                    else 'Task parameters: edit YAML and restart'))
        if "reward" in config and "weights" in config["reward"]:
            self.reward_weights.set(str(config["reward"]["weights"]))
        football = config.get("football", {})
        if football:
            self.reward_weights.set(str([
                football.get("robot_ball_weight", 1.0),
                football.get("ball_goal_weight", 1.0),
            ]))
            self.robot_position.set(str(football.get("robot_start_position", [0, 0, 0.7])))
            self.target_position.set(str(football.get("ball_start_position", [-2, 0, 5.2])))
            self.randomize_reset.set(bool(football.get("randomize_reset", True)))

        sim = config.get("sim", {})
        self.current_enabled.set(bool(sim.get("current", False)))
        self.current_uniform.set(bool(sim.get("current_uniform", True)))
        self.current_value.set(str(sim.get("current_value", [0.0, 0.0])))
        legacy_noise = bool(sim.get("enable_noise", False))
        self.offset_noise_enabled.set(bool(sim.get("enable_offset_noise", legacy_noise)))
        self.random_noise_enabled.set(bool(sim.get("enable_random_noise", legacy_noise)))

    def _runtime_params(self):
        params = {
            "reward_weights": _as_float_list(self.reward_weights.get()),
            "current_enabled": self.current_enabled.get(),
            "current_uniform": self.current_uniform.get(),
            "current_value": _as_float_list(self.current_value.get(), 2),
            "enable_offset_noise": self.offset_noise_enabled.get(),
            "enable_random_noise": self.random_noise_enabled.get(),
            "randomize_reset": self.randomize_reset.get(),
        }
        if not self.randomize_reset.get():
            params.update({
                "robot_start_position": _as_float_list(self.robot_position.get(), 3),
                "target_start_position": _as_float_list(self.target_position.get(), 3),
                "robot_start_rotation": _as_float_list(self.robot_rotation.get(), 3),
                "target_start_rotation": _as_float_list(self.target_rotation.get(), 3),
            })
        return params

    def start_env(self):
        if self.starting or self.env is not None:
            return
        self.starting = True
        try:
            self.stop_env()
            config = self._load_config()
            env_type = self.env_type.get()
            if env_type == "config":
                env_type = config.get("env", {}).get("type", "docking")
            # Fixed PID sends vehicle commands directly, not tuning parameters.
            if self.action_mode.get() == "fixed PID" and env_type in ("pid_tuning", "pid_rl_blend"):
                env_type = "docking"
            config.setdefault('env', {})['type'] = env_type
            if self.action_mode.get() == 'fixed PID' and not supports_fixed_pid(config):
                raise ValueError('Fixed PID is a docking controller. Select random, sliders or a trained model for this task.')
            config.setdefault("sim", {})["graphical_interface"] = self.graphical.get()
            if not config['env'].get('entry_point') and env_type.lower() in ("docking", "ds", "docking_ds", "pid_tuning", "pid_rl_blend"):
                reload_docking_environment()
            self.env = make_env_instance(env_type, 0, config)
            self.apply_params(show_error=False)
            self.reset_env()
            self._build_action_sliders()
            if self.action_mode.get() in ("fixed PID", "trained model"):
                self._policy_action()
            self.status.set("Environment started")
        except Exception as exc:
            self.stop_env()
            messagebox.showerror("Start failed", str(exc))
            self.status.set(f"Start failed: {exc}")
        finally:
            self.starting = False

    def apply_params(self, show_error=True):
        if not getattr(self, '_runtime_controls_supported', True):
            return
        try:
            params = self._runtime_params()
            if self.env is not None:
                self.env.unwrapped.update_runtime_params(params)
            self.status.set("Runtime parameters applied")
        except Exception as exc:
            if show_error:
                messagebox.showerror("Parameter error", str(exc))
            self.status.set(f"Parameter error: {exc}")

    def reset_env(self, clear_history=True):
        if self.env is None:
            return
        self.apply_params(show_error=False)
        self.obs, self.info = self.env.reset()
        self.last_action = None
        if isinstance(self.policy, PIDDockingPolicy):
            self.policy.reset()
        if clear_history:
            self.step_index = 0
            self.total_reward = 0.0
            self.clear_plot()
        self._update_plot_options()
        self._update_info_text(0.0, False, False)

    def _build_action_sliders(self):
        for child in self.action_slider_frame.winfo_children():
            child.destroy()
        self.action_sliders = []
        if self.env is None:
            return
        for index in range(int(self.env.action_size)):
            row = ttk.Frame(self.action_slider_frame)
            row.pack(fill=tk.X)
            ttk.Label(row, text=self._action_name(index), width=12).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            scale = ttk.Scale(row, from_=-1.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            value = ttk.Label(row, textvariable=var, width=8)
            value.pack(side=tk.LEFT)
            self.action_sliders.append(var)

    def _action(self):
        if self.env is None:
            return None
        mode = self.action_mode.get()
        if mode == "random":
            return self.env.action_space.sample()
        if mode == "manual":
            return np.array([var.get() for var in self.action_sliders], dtype=np.float32)
        if mode == "controller":
            if self.env.action_size in (1, 2, 16):
                raise ValueError("Use manual or trained model mode for blending/PID tuning; joystick axes are vehicle commands")
            return self._controller_action()
        if mode in ("fixed PID", "trained model"):
            return self._policy_action()
        return np.zeros(self.env.action_size, dtype=np.float32)

    def _action_name(self, index):
        names = getattr(self.env, 'policy_action_names', ())
        if index < len(names):
            return names[index]
        if self.env is not None and self.env.action_size in (1, 2):
            return f"K{index + 1}"
        if self.env is not None and self.env.action_size == 16:
            return f"{('kp', 'ki', 'kd', 'offset')[index // 4]}_{('x', 'y', 'z', 'yaw')[index % 4]}"
        return f"a{index}"

    def _policy_action(self):
        mode = self.action_mode.get()
        key = (mode, self.algorithm.get(), self.model_path.get().strip())
        if key != self.policy_key:
            if mode == "fixed PID":
                if self.env.action_size in (1, 2, 16):
                    raise ValueError("Restart in fixed PID mode to use direct docking commands")
                policy = PIDDockingPolicy(self.env.unwrapped, self.config.get("pid", {}))
            else:
                if not key[2] or key[2].upper() == "PID":
                    raise ValueError("Select a trained .zip or .onnx checkpoint for trained model mode")
                from library.evaluation import load_model
                policy = load_model(key[1], key[2], self.env, self.config)
            self.policy, self.policy_key = policy, key
        if mode == "fixed PID":
            action, _ = self.policy.predict(self.obs)
        else:
            from library.evaluation import predict_action
            action = predict_action(self.policy, key[1], self.obs)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.env.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Model output does not match the environment action space")
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def _controller_action(self):
        if self.controller is None:
            try:
                use_forces = bool(self.config.get("action", {}).get("force_6Dof", True))
                self.controller = LogitechController(
                    use_forces=use_forces,
                    action_size=self.env.action_size,
                )
                self.controller_error_shown = False
                self.status.set("Controller connected")
            except Exception as exc:
                if not self.controller_error_shown:
                    messagebox.showerror("Controller not available", str(exc))
                    self.controller_error_shown = True
                self.action_mode.set("zero")
                self.status.set(f"Controller unavailable: {exc}")
                return np.zeros(self.env.action_size, dtype=np.float32)

        action = np.asarray(self.controller.get_thruster_values(), dtype=np.float32).flatten()
        if len(action) != self.env.action_size:
            fixed = np.zeros(self.env.action_size, dtype=np.float32)
            fixed[: min(len(action), self.env.action_size)] = action[: self.env.action_size]
            return fixed
        return action

    def step_once(self, stop_on_episode_end=True):
        if self.env is None:
            self.status.set("Start an environment first")
            return False
        try:
            action = self._action()
            self.last_action = np.asarray(action, dtype=np.float32).flatten()
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            self.info = dict(self.info)
            if self.action_mode.get() == "fixed PID":
                pid = self.policy.controller
                self.info["pid_parameters"] = np.concatenate((pid.kp, pid.ki, pid.kd, pid.offsets)).tolist()
                self.info["pid_command"] = self.last_action.tolist()
            self.step_index += 1
            self.total_reward += float(reward)
            self._append_plot_values(float(reward))
            self._update_plot_options()
            self._update_info_text(float(reward), terminated, truncated)
            if terminated or truncated:
                if stop_on_episode_end:
                    self.running = False
                self.status.set(f"Episode finished: terminated={terminated}, truncated={truncated}")
                return True
            else:
                self.status.set(f"Step {self.step_index} reward={float(reward):.3f}")
                return False
        except Exception as exc:
            self.running = False
            messagebox.showerror("Step failed", str(exc))
            self.status.set(f"Step failed: {exc}")
            return False

    def resume(self):
        if self.env is None:
            self.status.set("Start an environment first")
            return
        if self.running:
            self.status.set("Already running")
            return
        self.running = True
        self._run_loop()

    def _run_loop(self):
        self.loop_after_id = None
        if not self.running:
            return
        episode_finished = self.step_once(stop_on_episode_end=False)
        if self.running and episode_finished:
            self.reset_env(clear_history=False)
            self.status.set("Episode ended; reset automatically for continuous resume")
        if self.running:
            self.loop_after_id = self.after(20, self._run_loop)

    def pause(self):
        self.running = False
        if self.loop_after_id is not None:
            self.after_cancel(self.loop_after_id)
            self.loop_after_id = None
        self.status.set("Paused")

    def stop_env(self):
        self.pause()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self.controller = None
        self.policy = None
        self.policy_key = None

    def new_plot(self):
        for curve in self.plot_curves:
            try:
                curve["line"].remove()
            except ValueError:
                pass
        self.plot_curves = []
        self.curve_listbox.delete(0, tk.END)
        self._refresh_plot()
        self.status.set("Plot reset. Add curves to start plotting.")

    def add_plot_curve(self):
        selection = self.plot_key.get()
        if not selection:
            return
        key = self._plot_key_from_selection(selection)
        label = selection
        if any(curve["key"] == key for curve in self.plot_curves):
            self.status.set(f"Curve already exists: {label}")
            return

        line, = self.ax.plot([], [], linewidth=1.8, label=label)
        self.plot_curves.append({
            "key": key,
            "label": label,
            "x": [],
            "y": [],
            "line": line,
        })
        self.curve_listbox.insert(tk.END, label)
        self._refresh_plot()
        self.status.set(f"Added curve: {label}")

    def remove_plot_curve(self):
        selection = self.curve_listbox.curselection()
        if not selection:
            self.status.set("Select a curve to remove.")
            return
        index = int(selection[0])
        curve = self.plot_curves.pop(index)
        curve["line"].remove()
        self.curve_listbox.delete(index)
        self._refresh_plot()
        self.status.set(f"Removed curve: {curve['label']}")

    def clear_plot(self):
        for curve in self.plot_curves:
            curve["x"].clear()
            curve["y"].clear()
            curve["line"].set_data([], [])
        self._refresh_plot()
        self.status.set("Plot data cleared.")

    def _refresh_plot(self):
        self.ax.relim()
        self.ax.autoscale_view()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        if self.plot_curves:
            self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def _append_plot_values(self, reward):
        for curve in self.plot_curves:
            value = self._resolve_plot_value(curve["key"], reward)
            if value is None or not np.isfinite(value):
                continue
            curve["x"].append(self.step_index)
            curve["y"].append(float(value))
            curve["line"].set_data(curve["x"], curve["y"])
        if self.plot_curves:
            self._refresh_plot()

    def _plot_key_from_selection(self, selection):
        return selection.split(" ", 1)[0]

    def _resolve_plot_value(self, key, reward):
        key = self._plot_key_from_selection(key)
        if key == "reward":
            return reward
        if key == "total_reward":
            return self.total_reward
        if key.startswith("obs[") and key.endswith("]"):
            index = int(key[4:-1])
            return float(self.obs[index]) if self.obs is not None and 0 <= index < len(self.obs) else None
        if key.startswith("action[") and key.endswith("]") and self.last_action is not None:
            index = int(key[7:-1])
            return float(self.last_action[index]) if 0 <= index < len(self.last_action) else None
        if key.startswith("info:"):
            field = key.split(":", 1)[1]
            if "[" in field and field.endswith("]"):
                name, index = field[:-1].split("[")
                value = self.info.get(name)
                if value is not None and int(index) < len(value):
                    return float(value[int(index)])
                return None
            return self._scalar(self.info.get(field))
        return None

    def _scalar(self, value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value).flatten()
            if arr.size:
                return float(arr[0])
        return None

    def _update_plot_options(self):
        values = ["reward", "total_reward"]
        if self.obs is not None:
            base_env = self.env.unwrapped
            obs_names = getattr(
                base_env,
                "total_observation_names",
                getattr(base_env, "observation_names", getattr(base_env, "state_names", [])),
            )
            for index in range(len(self.obs)):
                name = obs_names[index] if index < len(obs_names) else str(index)
                values.append(f"obs[{index}] {name}")
        if self.last_action is not None:
            values.extend([f"action[{i}] {self._action_name(i)}" for i in range(len(self.last_action))])
        if isinstance(self.info, dict):
            values.extend([f"info:{key}" for key in sorted(self.info)])
            for field in ("pid_parameters", "pid_command", "rl_command", "pid_rl_command", "upward_command", "blended_command"):
                vector = self.info.get(field, [])
                for index in range(len(vector)):
                    if field == "pid_parameters":
                        name = f"{('kp', 'ki', 'kd', 'offset')[index // 4]}_{('x', 'y', 'z', 'yaw')[index % 4]}"
                    else:
                        name = ("x", "y", "z", "yaw")[index] if len(vector) == 4 else ("x", "y", "z", "roll", "pitch", "yaw")[index]
                    values.append(f"info:{field}[{index}] {name}")
        self.plot_combo["values"] = values
        if self.plot_key.get() not in values:
            self.plot_key.set(values[0])

    def _update_info_text(self, reward, terminated, truncated):
        self.info_text.delete("1.0", tk.END)
        rows = [
            f"step: {self.step_index}",
            f"reward: {reward:.6f}",
            f"total_reward: {self.total_reward:.6f}",
            f"terminated: {terminated}",
            f"truncated: {truncated}",
            f"observation frame: {getattr(self.env.unwrapped, 'observation_frame', 'default') if self.env is not None else None}",
            f"action: {np.round(self.last_action, 3).tolist() if self.last_action is not None else None}",
            f"info: {self.info}",
        ]
        if "pid_parameters" in self.info:
            parameters = np.asarray(self.info["pid_parameters"]).reshape(4, 4)
            rows[6:6] = [f"PID {name} [x,y,z,yaw]: {np.round(values, 4).tolist()}"
                         for name, values in zip(("Kp", "Ki", "Kd", "offset"), parameters)]
        if "pid_command" in self.info:
            rows.insert(6, f"Vehicle command: {np.round(self.info['pid_command'], 4).tolist()}")
        self.info_text.insert(tk.END, "\n".join(rows))

