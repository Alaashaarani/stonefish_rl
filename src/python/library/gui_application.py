"""Single-window workspace for testing, evaluating and plotting."""

import tkinter as tk
from tkinter import ttk

from library.testing_gui import TestingGui
from library.compare_evaluation_gui import CompareEvaluationGui
from library.evaluation_plots import PlotEvaluationGui
from library.tasks_gui import TasksGui


class StonefishRLGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stonefish RL")
        self.geometry("1400x900")
        self.minsize(1000, 650)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.testing = TestingGui(self.tabs)
        self.evaluation = CompareEvaluationGui(self.tabs)
        self.plot_tab = ttk.Frame(self.tabs)
        self.plotting = PlotEvaluationGui(self.plot_tab)
        self.tasks = TasksGui(self.tabs, self.use_task_in_test, self.use_task_in_evaluation)
        self.tabs.add(self.tasks, text="Tasks")
        self.tabs.add(self.testing, text="Test")
        self.tabs.add(self.evaluation, text="Evaluate")
        self.tabs.add(self.plot_tab, text="Plot")
        self.protocol("WM_DELETE_WINDOW", self.close)

    def use_task_in_test(self, path):
        if self.testing.env is not None:
            self.tasks.status.set('Stop the active Test environment before selecting another task.')
            return
        self.testing.config_path.set(path)
        self.testing.env_type.set('config')
        self.testing.load_yaml()
        self.tabs.select(self.testing)

    def use_task_in_evaluation(self, path):
        if self.evaluation.process is not None:
            self.tasks.status.set('Wait for the active evaluation to finish before selecting another task.')
            return
        self.evaluation.config_path.set(path)
        self.evaluation.load_config(load_models=True)
        self.tabs.select(self.evaluation)

    def close(self):
        self.testing.stop_env()
        self.evaluation.stop_compare()
        self.destroy()

