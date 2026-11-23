"""GUI launcher for comparing multiple evaluation models."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from library.common import project_root, resolve_path
from library.gui_panel import GuiPanel
from library.configuration import load_config as load_workflow_config, resolve_config, DEFAULT_CONFIG


class CompareEvaluationGui(GuiPanel):
    """Build an evaluation-compare YAML and launch the existing evaluator."""

    def __init__(self, master=None):
        super().__init__(master, title="Stonefish RL Compare Evaluation GUI", geometry="1120x780")

        self.config_path = tk.StringVar(value=DEFAULT_CONFIG)
        self.output_dir = tk.StringVar(value="evaluation_results")
        self.num_episodes = tk.IntVar(value=5)
        self.step_per_print = tk.IntVar(value=0)
        self.parallel = tk.BooleanVar(value=False)
        self.max_workers = tk.IntVar(value=1)
        self.quiet = tk.BooleanVar(value=False)
        self.force_headless_parallel = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Idle")

        self.model_rows = []
        self.process = None
        self.error_lines = []
        self.output_queue = queue.Queue()
        self.output_after_id = None
        self._build_layout()
        self.load_config(load_models=True)

    def _build_layout(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(root, text="Evaluation Config", padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="YAML").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.config_path, width=70).grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(top, text="Browse", command=self._browse_config).grid(row=0, column=2)
        ttk.Button(top, text="Load", command=self.load_config).grid(row=0, column=3)

        ttk.Label(top, text="Output dir").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.output_dir, width=70).grid(row=1, column=1, sticky=tk.EW)
        ttk.Button(top, text="Browse", command=self._browse_output_dir).grid(row=1, column=2)
        top.columnconfigure(1, weight=1)

        middle = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, pady=8)

        left = ttk.Frame(middle)
        right = ttk.Frame(middle)
        middle.add(left, weight=1)
        middle.add(right, weight=1)

        self._build_model_panel(left)
        self._build_yaml_panel(right)
        self._build_run_panel(root)

    def _build_model_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Models", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Select", width=7).pack(side=tk.LEFT)
        ttk.Label(header, text="Algorithm", width=12).pack(side=tk.LEFT)
        ttk.Label(header, text="Model path").pack(side=tk.LEFT)

        self.rows_frame = ttk.Frame(frame)
        self.rows_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Add Model", command=self.add_model_row).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Load Directory", command=self.load_models_from_directory).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Select All", command=lambda: self.set_all_model_rows(True)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Unselect All", command=lambda: self.set_all_model_rows(False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Remove Selected", command=self.remove_model_row).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Load From YAML", command=self._rows_from_yaml).pack(side=tk.LEFT)

    def _build_yaml_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Other YAML Parameters", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        self.yaml_text = tk.Text(frame, wrap=tk.NONE)
        self.yaml_text.pack(fill=tk.BOTH, expand=True)

    def _build_run_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Run", padding=8)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="Episodes/model").pack(side=tk.LEFT)
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.num_episodes, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(frame, text="Step print").pack(side=tk.LEFT)
        ttk.Spinbox(frame, from_=0, to=100000, textvariable=self.step_per_print, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(frame, text="Parallel", variable=self.parallel).pack(side=tk.LEFT, padx=8)
        ttk.Label(frame, text="Workers").pack(side=tk.LEFT)
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.max_workers, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(frame, text="Quiet", variable=self.quiet).pack(side=tk.LEFT)
        ttk.Checkbutton(
            frame,
            text="Headless parallel",
            variable=self.force_headless_parallel,
        ).pack(side=tk.LEFT, padx=8)

        commands = ttk.Frame(parent)
        commands.pack(fill=tk.X)
        ttk.Button(commands, text="Save YAML", command=self.save_compare_yaml).pack(side=tk.LEFT, padx=4)
        ttk.Button(commands, text="Run Compare", command=self.run_compare).pack(side=tk.LEFT)
        ttk.Button(commands, text="Stop", command=self.stop_compare).pack(side=tk.LEFT, padx=4)

        ttk.Label(parent, textvariable=self.status, anchor=tk.W).pack(fill=tk.X)
        self.log_text = tk.Text(parent, height=12)
        self.log_text.pack(fill=tk.X, pady=(8, 0))

    def add_model_row(self, algorithm="PPO", model_path="", selected=True, evaluation_name="", note=""):
        row = ttk.Frame(self.rows_frame)
        row.pack(fill=tk.X, pady=2)
        metadata = ttk.Frame(row)
        metadata.pack(side=tk.BOTTOM, fill=tk.X)
        selected_var = tk.BooleanVar(value=bool(selected))
        alg_var = tk.StringVar(value=algorithm)
        path_var = tk.StringVar(value=model_path)
        ttk.Checkbutton(row, variable=selected_var).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=alg_var,
            values=["PPO", "SAC", "TD3", "ONNX", "PID"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="Browse", command=lambda: self._browse_model(path_var)).pack(side=tk.LEFT)
        row.evaluation_name = tk.StringVar(value=evaluation_name)
        row.note = tk.StringVar(value=note)
        ttk.Label(metadata, text="Name").pack(side=tk.LEFT)
        ttk.Entry(metadata, textvariable=row.evaluation_name, width=18).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(metadata, text="Note").pack(side=tk.LEFT)
        ttk.Entry(metadata, textvariable=row.note, width=18).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.model_rows.append((row, selected_var, alg_var, path_var))

    def set_all_model_rows(self, selected):
        for _, selected_var, _, _ in self.model_rows:
            selected_var.set(bool(selected))
        self.status.set("Selected all models." if selected else "Unselected all models.")

    def remove_model_row(self):
        if not self.model_rows:
            return
        selected_rows = [
            item for item in self.model_rows
            if bool(item[1].get())
        ]
        if not selected_rows:
            selected_rows = [self.model_rows[-1]]

        for item in selected_rows:
            item[0].destroy()
            self.model_rows.remove(item)

        if not self.model_rows:
            self.add_model_row()

    def infer_algorithm(self, model_path):
        path = Path(model_path)
        if path.suffix.lower() == ".onnx":
            return "ONNX"
        prefix = path.stem.split("_", 1)[0].upper()
        if prefix in {"PPO", "SAC", "TD3"}:
            return prefix
        return "PPO"

    def load_models_from_directory(self):
        directory = filedialog.askdirectory(initialdir=resolve_path("models"))
        if not directory:
            return

        model_files = sorted(
            path for path in Path(directory).iterdir()
            if path.is_file() and path.suffix.lower() in {".zip", ".onnx"}
        )
        if not model_files:
            messagebox.showinfo("No models found", "No .zip or .onnx model files were found in this directory.")
            return

        for row, _, _, _ in self.model_rows:
            row.destroy()
        self.model_rows = []

        for path in model_files:
            self.add_model_row(self.infer_algorithm(path), str(path), selected=True)
        self.status.set(f"Loaded {len(model_files)} models from {directory}")

    def _browse_config(self):
        path = filedialog.askopenfilename(
            initialdir=resolve_path("include/parameters"),
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.config_path.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(initialdir=project_root())
        if path:
            self.output_dir.set(path)

    def _browse_model(self, path_var):
        path = filedialog.askopenfilename(
            initialdir=resolve_path("models"),
            filetypes=[("Models", "*.zip *.onnx"), ("All files", "*.*")],
        )
        if path:
            path_var.set(path)

    def load_config(self, *, load_models=False):
        try:
            config = load_workflow_config(self.config_path.get(), "evaluation")
            self.yaml_text.delete("1.0", tk.END)
            self.yaml_text.insert(tk.END, yaml.safe_dump(config, sort_keys=False))
            eval_cfg = config.get("evaluate", {})
            compare_cfg = config.get("compare", {})
            self.num_episodes.set(int(compare_cfg.get("num_episodes", eval_cfg.get("num_episodes", 5))))
            self.step_per_print.set(int(compare_cfg.get("step_per_print", eval_cfg.get("step_per_print", 0))))
            self.parallel.set(bool(compare_cfg.get("parallel", False)))
            self.max_workers.set(int(compare_cfg.get("max_workers", config.get("env", {}).get("instances", 1))))
            self.force_headless_parallel.set(bool(compare_cfg.get("force_headless_parallel", True)))
            if load_models:
                self._rows_from_config(config)
            self.status.set("Loaded config")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            self.status.set(f"Load failed: {exc}")

    def _rows_from_yaml(self):
        try:
            self._rows_from_config(self._text_config())
        except Exception as exc:
            messagebox.showerror("YAML error", str(exc))

    def _rows_from_config(self, config):
        for row, _, _, _ in self.model_rows:
            row.destroy()
        self.model_rows = []

        compare_cfg = config.get("compare", {}) or {}
        algorithms = compare_cfg.get("algorithms", [])
        model_paths = compare_cfg.get("model_paths", [])
        if not model_paths and config.get("evaluate"):
            algorithms = [config["evaluate"].get("algorithm", "PPO")]
            model_paths = [config["evaluate"].get("model_path", "")]

        metadata = compare_cfg if compare_cfg.get("model_paths") else config.get("evaluate", {})
        for index, (alg, path) in enumerate(zip(algorithms, model_paths)):
            values = {}
            for key in ("evaluation_name", "note"):
                value = metadata.get(key, "")
                if isinstance(value, list):
                    value = value[0] if len(value) == 1 else value[index]
                values[key] = value or ""
            self.add_model_row(str(alg).upper(), str(path), selected=True, **values)

        if not self.model_rows:
            self.add_model_row()

    def _text_config(self):
        return resolve_config(yaml.safe_load(self.yaml_text.get("1.0", tk.END)) or {}, "evaluation")

    def _model_specs(self):
        algorithms = []
        model_paths = []
        for _, selected_var, alg_var, path_var in self.model_rows:
            if not selected_var.get():
                continue
            path = path_var.get().strip() or ("PID" if alg_var.get() == "PID" else "")
            if not path:
                continue
            algorithms.append(alg_var.get().upper())
            model_paths.append(path)
        if not model_paths:
            raise ValueError("Add at least one model path.")
        return algorithms, model_paths

    def _build_compare_config(self):
        config = self._text_config()
        algorithms, model_paths = self._model_specs()
        compare_cfg = config.setdefault("compare", {})
        workers = self.max_workers.get()
        if workers < 1:
            raise ValueError("Workers must be a positive integer.")
        compare_cfg.update({
            "compare": True,
            "num_models": len(model_paths),
            "algorithms": algorithms,
            "model_paths": model_paths,
            "num_episodes": int(self.num_episodes.get()),
            "step_per_print": int(self.step_per_print.get()),
            "parallel": bool(self.parallel.get()),
            "max_workers": workers,
            "batch_size": 20,
            "force_headless_parallel": bool(self.force_headless_parallel.get()),
        })
        selected_rows = [row for row, selected, alg, path in self.model_rows
                         if selected.get() and (path.get().strip() or alg.get() == "PID")]
        compare_cfg["evaluation_name"] = [row.evaluation_name.get().strip() for row in selected_rows]
        compare_cfg["note"] = [row.note.get() for row in selected_rows]
        for key in ["yaw_observation", "history_length", "evaluation_tag", "evaluation_tags"]:
            value = compare_cfg.get(key)
            if isinstance(value, list) and len(value) not in (1, len(model_paths)):
                compare_cfg.pop(key, None)
        return config

    def save_compare_yaml(self):
        try:
            path = filedialog.asksaveasfilename(
                initialdir=resolve_path("include/parameters"),
                initialfile="evaluation_compare_gui.yaml",
                defaultextension=".yaml",
                filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
            )
            if not path:
                return None
            config = self._build_compare_config()
            with open(path, "w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, sort_keys=False)
            self.status.set(f"Saved {path}")
            return path
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            self.status.set(f"Save failed: {exc}")
            return None

    def run_compare(self):
        if self.process is not None:
            messagebox.showwarning("Already running", "An evaluation process is already running.")
            return
        try:
            config = self._build_compare_config()
            compare = config["compare"]
            workers = min(compare["max_workers"], compare["num_models"], compare["batch_size"]) if compare["parallel"] else 1
            if compare["parallel"] and workers == 1 and compare["num_models"] > 1:
                messagebox.showwarning("One evaluation worker", "Parallel is enabled, but Workers is 1. Increase Workers to evaluate models concurrently.")
                return
            output_dir = self.output_dir.get().strip() or "evaluation_results"
            Path(resolve_path(output_dir)).mkdir(parents=True, exist_ok=True)

            generated_dir = Path(resolve_path("include/parameters/generated_gui"))
            generated_dir.mkdir(parents=True, exist_ok=True)
            fd, config_path = tempfile.mkstemp(
                prefix="evaluation_compare_",
                suffix=".yaml",
                dir=str(generated_dir),
                text=True,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, sort_keys=False)

            cmd = [
                sys.executable,
                "-m",
                "stonefish_rl_evaluation",
                config_path,
                "--output-dir",
                output_dir,
            ]
            cmd.append("--parallel" if self.parallel.get() else "--sequential")
            if self.quiet.get():
                cmd.append("--quiet")

            env = os.environ.copy()
            src_python = str(Path(project_root()) / "src" / "python")
            env["PYTHONPATH"] = src_python + os.pathsep + env.get("PYTHONPATH", "")

            self.log_text.delete("1.0", tk.END)
            self.error_lines = []
            self._log("$ " + " ".join(cmd))
            self._log(f"Evaluation workers: {workers} | Models: {compare['num_models']}")
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.status.set("Evaluation running")
            threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()
            self.output_after_id = self.after(100, self._poll_process_output)
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))
            self.status.set(f"Run failed: {exc}")

    def _read_process_output(self, process):
        # The worker only queues data; Tk widgets belong to the main thread.
        for line in process.stdout:
            self.output_queue.put(("line", line.rstrip()))
        process.stdout.close()
        self.output_queue.put(("done", process.wait()))

    def _poll_process_output(self):
        self.output_after_id = None
        for _ in range(200):
            try:
                kind, value = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._log(value)
                if "ERROR while evaluating" in value:
                    self.error_lines.append(value)
            else:
                self.process = None
                self.status.set(f"Evaluation finished with code {value}")
                if self.error_lines:
                    messagebox.showwarning("Evaluation errors", "\n".join(self.error_lines[-8:]), parent=self.window)
        if self.process is not None:
            self.output_after_id = self.after(100, self._poll_process_output)

    def destroy(self):
        if self.output_after_id is not None:
            self.after_cancel(self.output_after_id)
            self.output_after_id = None
        super().destroy()

    def _log(self, line):
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)

    def stop_compare(self):
        if self.process is None:
            return
        self.process.terminate()
        self.status.set("Stopping evaluation")

