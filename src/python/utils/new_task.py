"""Create an editable reaching-task example without modifying the core registry."""

if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.

import argparse
import importlib.util
from copy import deepcopy
from pathlib import Path
import re

import yaml

from library.common import project_root


def create_task(name, root=None):
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)*', name):
        raise ValueError('Use a lowercase task name such as waypoint_following.')
    root = Path(root or project_root())
    class_name = ''.join(word.capitalize() for word in name.split('_')) + 'Env'
    package = root / 'src/python/library'
    source = package / (name + '.py')
    config_path = root / 'include/parameters/tasks' / (name + '.yaml')
    if source.exists() or config_path.exists():
        raise ValueError('Task files already exist. Choose a new name; existing files are never overwritten.')
    if importlib.util.find_spec(name) is not None or importlib.util.find_spec('library.' + name) is not None:
        raise ValueError('That name already belongs to an importable Python module. Choose a task-specific name.')
    template_root = Path(project_root())
    template = (template_root / 'src/python/library/reach_target_env.py').read_text()
    template = template.replace('class ReachTargetEnv(', 'class ' + class_name + '(')
    template = template.replace('ReachTargetEnv reward.weights', class_name + ' reward.weights')
    config = deepcopy(yaml.safe_load((template_root / 'include/parameters/tasks/reach_target.yaml').read_text()))
    config['common']['env'].update(type=name, entry_point='library.%s:%s' % (name, class_name),
                                  reward_names=['distance', 'effort', 'success_bonus'])
    config['common']['evaluate']['model_path'] = 'models/%s/CHOOSE_A_CHECKPOINT.zip' % name
    config['training']['train'].update(save_name=name, save_path='models/' + name)
    config['training']['log'].update(log_dir='logs/' + name, tags=[name])
    package.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with source.open('x') as stream:
        stream.write('# Task starter: change the reward, success condition and observations for your task.\n' + template)
    with config_path.open('x') as stream:
        stream.write('# Editable task example. Validate and test before starting training.\n' + yaml.safe_dump(config, sort_keys=False))
    return source, config_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='New lowercase task name, for example waypoint_following.')
    args = parser.parse_args(argv)
    try:
        source, config = create_task(args.name)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print('Created task: %s\nCreated YAML: %s' % (source, config))
    print('Next: python3 src/python/utils/check_task.py include/parameters/tasks/%s.yaml' % args.name)
    print('This is a reaching-task starter, not an implemented policy for the name you chose.')


if __name__ == '__main__':
    main()
