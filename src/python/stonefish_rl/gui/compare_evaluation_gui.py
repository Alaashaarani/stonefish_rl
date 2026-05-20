"""GUI launcher for comparing multiple evaluation models."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from stonefish_rl.utils.utils import project_root, resolve_path


class CompareEvaluationGui(tk.Tk):
    """Build an evaluation-compare YAML and launch the existing evaluator."""

    def __init__(self):
        super().__init__()
        self.title("Stonefish RL Compare Evaluation GUI")
        self.geometry("1120x780")

        self.config_path = tk.StringVar(value="include/parameters/evaluation_param.yaml")
        self.output_dir = tk.StringVar(value="evaluation_results")
        self.num_episodes = tk.IntVar(value=5)
        self.step_per_print = tk.IntVar(value=0)
        self.parallel = tk.BooleanVar(value=False)
        self.quiet = tk.BooleanVar(value=False)
        self.force_headless_parallel = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Idle")

        self.model_rows = []
        self.process = None
        self._build_layout()
        self.load_config()

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
        ttk.Label(header, text="Algorithm", width=12).pack(side=tk.LEFT)
        ttk.Label(header, text="Model path").pack(side=tk.LEFT)

        self.rows_frame = ttk.Frame(frame)
        self.rows_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Add Model", command=self.add_model_row).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Remove Last", command=self.remove_model_row).pack(side=tk.LEFT, padx=4)
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
        ttk.Checkbutton(frame, text="Quiet", variable=self.quiet).pack(side=tk.LEFT)
        ttk.Checkbutton(
            frame,
            text="Headless parallel",
            variable=self.force_headless_parallel,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(frame, text="Save YAML", command=self.save_compare_yaml).pack(side=tk.LEFT, padx=4)
        ttk.Button(frame, text="Run Compare", command=self.run_compare).pack(side=tk.LEFT)
        ttk.Button(frame, text="Stop", command=self.stop_compare).pack(side=tk.LEFT, padx=4)

        ttk.Label(parent, textvariable=self.status, anchor=tk.W).pack(fill=tk.X)
        self.log_text = tk.Text(parent, height=12)
        self.log_text.pack(fill=tk.X, pady=(8, 0))

    def add_model_row(self, algorithm="PPO", model_path=""):
        row = ttk.Frame(self.rows_frame)
        row.pack(fill=tk.X, pady=2)
        alg_var = tk.StringVar(value=algorithm)
        path_var = tk.StringVar(value=model_path)
        ttk.Combobox(
            row,
            textvariable=alg_var,
            values=["PPO", "SAC", "TD3", "ONNX"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="Browse", command=lambda: self._browse_model(path_var)).pack(side=tk.LEFT)
        self.model_rows.append((row, alg_var, path_var))

    def remove_model_row(self):
        if not self.model_rows:
            return
        row, _, _ = self.model_rows.pop()
        row.destroy()

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

    def load_config(self):
        try:
            with open(resolve_path(self.config_path.get()), "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            self.yaml_text.delete("1.0", tk.END)
            self.yaml_text.insert(tk.END, yaml.safe_dump(config, sort_keys=False))
            eval_cfg = config.get("evaluate", {})
            compare_cfg = config.get("compare", {})
            self.num_episodes.set(int(compare_cfg.get("num_episodes", eval_cfg.get("num_episodes", 5))))
            self.step_per_print.set(int(compare_cfg.get("step_per_print", eval_cfg.get("step_per_print", 0))))
            self.parallel.set(bool(compare_cfg.get("parallel", False)))
            self.force_headless_parallel.set(bool(compare_cfg.get("force_headless_parallel", True)))
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
        for row, _, _ in self.model_rows:
            row.destroy()
        self.model_rows = []

        compare_cfg = config.get("compare", {}) or {}
        algorithms = compare_cfg.get("algorithms", [])
        model_paths = compare_cfg.get("model_paths", [])
        if not model_paths and config.get("evaluate"):
            algorithms = [config["evaluate"].get("algorithm", "PPO")]
            model_paths = [config["evaluate"].get("model_path", "")]

        for alg, path in zip(algorithms, model_paths):
            self.add_model_row(str(alg).upper(), str(path))

        if not self.model_rows:
            self.add_model_row()

    def _text_config(self):
        return yaml.safe_load(self.yaml_text.get("1.0", tk.END)) or {}

    def _model_specs(self):
        algorithms = []
        model_paths = []
        for _, alg_var, path_var in self.model_rows:
            path = path_var.get().strip()
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
        compare_cfg.update({
            "compare": True,
            "num_models": len(model_paths),
            "algorithms": algorithms,
            "model_paths": model_paths,
            "num_episodes": int(self.num_episodes.get()),
            "step_per_print": int(self.step_per_print.get()),
            "parallel": bool(self.parallel.get()),
            "force_headless_parallel": bool(self.force_headless_parallel.get()),
        })
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
                "stonefish_rl.evaluation.evaluate_docking_compare",
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
            self._log("$ " + " ".join(cmd))
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
            threading.Thread(target=self._read_process_output, daemon=True).start()
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))
            self.status.set(f"Run failed: {exc}")

    def _read_process_output(self):
        assert self.process is not None
        for line in self.process.stdout:
            self.after(0, self._log, line.rstrip())
        code = self.process.wait()
        self.process = None
        self.after(0, self.status.set, f"Evaluation finished with code {code}")

    def _log(self, line):
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)

    def stop_compare(self):
        if self.process is None:
            return
        self.process.terminate()
        self.status.set("Stopping evaluation")


def main():
    app = CompareEvaluationGui()
    app.protocol("WM_DELETE_WINDOW", lambda: (app.stop_compare(), app.destroy()))
    app.mainloop()


if __name__ == "__main__":
    main()
