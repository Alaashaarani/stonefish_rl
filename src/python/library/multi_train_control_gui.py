"""Small GUI to gracefully stop the active multi-training run."""

from __future__ import annotations

import tkinter as tk
import argparse
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from library.multitraining import ACTIVE_STATUS_NAME
from library.common import resolve_path


class MultiTrainControlGui(tk.Tk):
    """Monitor multi-training and request a clean stop for the current run."""

    def __init__(self, control_dir=None):
        super().__init__()
        self.title("Stonefish RL Multi-Train Control")
        self.geometry("760x420")

        self.control_dir = tk.StringVar(value=str(control_dir) if control_dir else "")
        self.status_text = tk.StringVar(value="Select a multi-train control directory.")
        self.active_status = {}
        self.refresh_after_id = None

        self._build_layout()
        self._schedule_refresh()

    def _build_layout(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(root, text="Control Directory", padding=8)
        top.pack(fill=tk.X)
        ttk.Entry(top, textvariable=self.control_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(top, text="Browse", command=self.browse_control_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Use Latest", command=self.use_latest_control_dir).pack(side=tk.LEFT)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=8)
        ttk.Button(actions, text="Refresh", command=self.manual_refresh).pack(side=tk.LEFT)
        ttk.Button(actions, text="Stop Current Training", command=self.stop_current_training).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Label(root, textvariable=self.status_text, anchor=tk.W).pack(fill=tk.X)

        self.details = tk.Text(root, height=14)
        self.details.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def browse_control_dir(self):
        initial = resolve_path("include/parameters/generated_multi_train")
        path = filedialog.askdirectory(initialdir=initial)
        if path:
            self.control_dir.set(path)
            self.refresh_status()

    def use_latest_control_dir(self):
        root = Path(resolve_path("include/parameters/generated_multi_train"))
        candidates = sorted(
            root.glob("*/control"),
            key=lambda path: (path / ACTIVE_STATUS_NAME).stat().st_mtime
            if (path / ACTIVE_STATUS_NAME).exists()
            else path.stat().st_mtime,
        )
        if not candidates:
            messagebox.showinfo("No control directory", f"No control directory found under {root}")
            return
        self.control_dir.set(str(candidates[-1]))
        self.refresh_status()

    def status_path(self):
        control_dir_value = self.control_dir.get().strip()
        if not control_dir_value:
            return None
        control_dir = Path(control_dir_value).expanduser()
        return control_dir / ACTIVE_STATUS_NAME

    def _schedule_refresh(self):
        if self.refresh_after_id is None:
            self.refresh_after_id = self.after(1000, self.refresh_status)

    def manual_refresh(self):
        if self.refresh_after_id is not None:
            self.after_cancel(self.refresh_after_id)
            self.refresh_after_id = None
        self.refresh_status()

    def refresh_status(self):
        self.refresh_after_id = None
        try:
            status_path = self.status_path()
            if status_path is None or not status_path.exists():
                self.active_status = {}
                self.status_text.set("Waiting for active status file...")
                self._write_details({})
            else:
                with status_path.open("r") as file:
                    self.active_status = yaml.safe_load(file) or {}
                status = self.active_status.get("status", "unknown")
                run_number = self.active_status.get("run_number", "-")
                run_count = self.active_status.get("run_count", "-")
                self.status_text.set(f"Status: {status} | Run {run_number}/{run_count}")
                self._write_details(self.active_status)
        finally:
            self._schedule_refresh()

    def _write_details(self, data):
        self.details.delete("1.0", tk.END)
        if data:
            self.details.insert(tk.END, yaml.safe_dump(data, sort_keys=False))
        else:
            self.details.insert(
                tk.END,
                "Waiting for multi-training to write its active status file.\n"
                "If training was not started, run: python3 src/python/stonefish_rl_training.py multi\n"
                "Example: include/parameters/generated_multi_train/<timestamp>/control\n",
            )

    def stop_current_training(self):
        if not self.active_status:
            messagebox.showwarning("No active run", "No active multi-training status was found.")
            return

        stop_file = self.active_status.get("stop_file")
        if not stop_file:
            messagebox.showwarning("No stop file", "The active status does not contain a stop_file path.")
            return

        stop_path = Path(stop_file).expanduser()
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        stop_path.write_text("stop requested from GUI\n")
        self.status_text.set(f"Stop requested: {stop_path}")
        messagebox.showinfo(
            "Stop requested",
            "The current training will stop cleanly at the next callback check, "
            "then multi-train will continue with the next run.",
        )

