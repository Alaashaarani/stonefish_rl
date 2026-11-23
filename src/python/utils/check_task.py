"""Validate a task without launching a simulator."""

if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.

import argparse
import json
import yaml
from library.configuration import load_config, DEFAULT_CONFIG
from library.task_check import check_config

def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate a task YAML without launching simulators or loading model weights.')
    parser.add_argument('config', nargs='?', default=DEFAULT_CONFIG)
    parser.add_argument('--mode', choices=('training', 'evaluation', 'testing'), default='training')
    parser.add_argument('--show-config', action='store_true', help='Print the effective merged configuration.')
    parser.add_argument('--json', action='store_true', help='Print a machine-readable check report.')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.mode)
        report = check_config(config, args.mode)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        report = dict(ok=False, details={}, errors=[str(exc)], warnings=[])
        config = None
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print('Task configuration: ' + ('PASS' if report['ok'] else 'NEEDS ATTENTION'))
        print(yaml.safe_dump(report['details'], sort_keys=False).strip())
        for level in ('errors', 'warnings'):
            for message in report[level]:
                print('%s: %s' % (level[:-1].upper(), message))
        print('No simulators started. Scene contents and physical behavior require a live test.')
    if args.show_config and config is not None:
        print(yaml.safe_dump(config, sort_keys=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
