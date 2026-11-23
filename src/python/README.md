# Python code

```text
src/python/
├── stonefish_rl_training.py    # single training, sweeps and stage sequences
├── stonefish_rl_evaluation.py  # evaluate and compare policies
├── stonefish_rl_gui.py         # Tasks / Test / Evaluate / Plot workspace
├── library/                   # reusable classes and implementation
└── utils/                     # task checks, benchmarks, plots and other tools
```

The three executables own their command-line interfaces. Shared code lives in
one flat `library/` package. Tools may use the library; the library does not
depend on the tools.

From the repository root:

```bash
python3 src/python/stonefish_rl_gui.py
python3 src/python/stonefish_rl_training.py single --check
python3 src/python/stonefish_rl_training.py multi --dry-run --no-gui
python3 src/python/stonefish_rl_evaluation.py --help
python3 src/python/utils/check_task.py include/parameters/tasks/reach_target.yaml
python3 src/python/utils/benchmark_instances.py --help
```

Use one training executable with `single` or `multi`. For example:

```bash
python3 src/python/stonefish_rl_training.py single include/parameters/tasks/reach_target.yaml
python3 src/python/stonefish_rl_training.py multi include/parameters/multi_train.yaml
```

| Edit | Library module |
| --- | --- |
| Simulator communication | `library/env_base.py` |
| Docking, football, reaching behavior | `library/docking_env.py`, `library/football_env.py`, `library/reach_target_env.py` |
| Common task lifecycle and selection | `library/task_env.py`, `library/env_factory.py` |
| Configuration and paths | `library/configuration.py`, `library/common.py` |
| Training and stage orchestration | `library/training.py`, `library/multitraining.py` |
| GUI classes | `library/gui_application.py`, `library/tasks_gui.py`, `library/testing_gui.py`, `library/compare_evaluation_gui.py` |

Imports use package names, for example `from library.task_env import TaskStep`.
Set `PYTHONPATH` to `src/python` when importing from outside this directory.
Tool scripts also support direct execution from another working directory.

`utils/new_task.py my_task` creates `library/my_task.py` and a task YAML using
`env.entry_point: library.my_task:MyTaskEnv`. See
[the task guide](../../docs/adding_tasks.md) for implementation and testing.

Model artifacts belong in `models/` and plots in `evaluation_results/`, outside
the source tree. Python creates `__pycache__/` directories automatically.
