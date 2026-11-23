"""Choose a task and inspect its effective settings without starting a simulator."""
import shlex
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

import yaml

from library.configuration import DEFAULT_CONFIG, load_config
from library.task_check import check_config
from library.common import resolve_path


EXAMPLES = {
    'Current docking experiment': DEFAULT_CONFIG,
    'Football example': 'include/parameters/tasks/football.yaml',
    'Team football: solo pretraining': 'include/parameters/tasks/team_football_solo.yaml',
    'Reach target example': 'include/parameters/tasks/reach_target.yaml',
}


class TasksGui(ttk.Frame):
    def __init__(self, master, use_in_test=None, use_in_evaluation=None):
        super().__init__(master, padding=16)
        self.config_path = tk.StringVar(value=DEFAULT_CONFIG)
        self.mode = tk.StringVar(value='training')
        self.example = tk.StringVar(value='Current docking experiment')
        self.status = tk.StringVar(value='Select a task YAML, then check it. No simulation starts from this tab.')
        ttk.Label(self, text='Task workspace', font=('TkDefaultFont', 16, 'bold')).pack(anchor=tk.W)
        ttk.Label(self, text='A task combines a scene, observations, actions, reward and success condition.').pack(anchor=tk.W, pady=(4, 12))
        row = ttk.Frame(self)
        row.pack(fill=tk.X)
        ttk.Label(row, text='Examples').pack(side=tk.LEFT)
        examples = ttk.Combobox(row, textvariable=self.example, values=list(EXAMPLES), state='readonly', width=30)
        examples.pack(side=tk.LEFT, padx=8)
        examples.bind('<<ComboboxSelected>>', self.select_example)
        ttk.Label(row, text='Workflow').pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(row, textvariable=self.mode, values=('training', 'testing', 'evaluation'), state='readonly', width=14).pack(side=tk.LEFT)
        paths = ttk.Frame(self)
        paths.pack(fill=tk.X, pady=10)
        ttk.Entry(paths, textvariable=self.config_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(paths, text='Browse YAML', command=self.browse).pack(side=tk.LEFT, padx=6)
        actions = ttk.Frame(self)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text='Check task', command=self.check).pack(side=tk.LEFT)
        ttk.Button(actions, text='Show effective YAML', command=self.show_config).pack(side=tk.LEFT, padx=6)
        if use_in_test:
            ttk.Button(actions, text='Use in Test', command=lambda: use_in_test(self.config_path.get())).pack(side=tk.LEFT, padx=6)
        if use_in_evaluation:
            ttk.Button(actions, text='Use in Evaluate', command=lambda: use_in_evaluation(self.config_path.get())).pack(side=tk.LEFT, padx=6)
        ttk.Label(self, textvariable=self.status, wraplength=1100).pack(fill=tk.X, pady=12)
        self.output = ScrolledText(self, wrap=tk.WORD, height=24)
        self.output.pack(fill=tk.BOTH, expand=True)
        self.write('Adding your own task:\n\n'
                   '1. Start with include/parameters/tasks/reach_target.yaml.\n'
                   '2. Subclass StonefishTaskEnv and implement evaluate_transition().\n'
                   '3. Set env.entry_point to your_package.your_module:YourTask.\n'
                   '4. Run Check task, then test one instance before training.\n\n'
                   'Guide: docs/adding_tasks.md\n'
                   'Existing docking checkpoints require their original observation/action layout.\n')

    def write(self, text):
        self.output.configure(state=tk.NORMAL)
        self.output.delete('1.0', tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state=tk.DISABLED)

    def select_example(self, event=None):
        self.config_path.set(EXAMPLES[self.example.get()])
        self.status.set('Example selected. Check the settings before launching it from another tab.')

    def browse(self):
        path = filedialog.askopenfilename(initialdir=resolve_path('include/parameters'), filetypes=[('Task YAML', '*.yaml *.yml')])
        if path:
            self.config_path.set(path)

    def check(self):
        try:
            config = load_config(self.config_path.get(), self.mode.get())
            report = check_config(config, self.mode.get())
            self.status.set('Configuration checks passed; no simulator was started.' if report['ok']
                            else 'Fix the errors below before launching this task.')
            command = 'python3 src/python/utils/check_task.py %s --mode %s' % (
                shlex.quote(self.config_path.get()), self.mode.get())
            self.write(yaml.safe_dump(report, sort_keys=False) + '\nTerminal check:\n' + command +
                       '\n\nThis checks configuration and files. Scene contents and physical behavior need a live test.\n')
        except Exception as exc:
            self.status.set('Could not check the task.')
            self.write(str(exc))

    def show_config(self):
        try:
            self.write(yaml.safe_dump(load_config(self.config_path.get(), self.mode.get()), sort_keys=False))
            self.status.set('Effective %s settings after merging common and workflow overrides.' % self.mode.get())
        except Exception as exc:
            self.status.set('Could not load the task.')
            self.write(str(exc))
