"""Train a single task or a multi-run/multi-stage experiment from one CLI."""
import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    single = modes.add_parser('single', help='Train one task using its training YAML section.')
    single.add_argument('config', nargs='?', default='include/parameters/stonefish_rl.yaml')
    single.add_argument('--check', action='store_true', help='Check configuration without starting training.')
    multi = modes.add_parser('multi', help='Run a sweep or sequential curriculum from a multi-training YAML.')
    root = Path(__file__).resolve().parents[2]
    multi.add_argument('multi_config', nargs='?', default=str(root / 'include/parameters/multi_train.yaml'))
    multi.add_argument('--base-config', help='Override the base training YAML.')
    multi.add_argument('--output-dir', default=str(root / 'include/parameters/generated_multi_train'))
    multi.add_argument('--dry-run', action='store_true', help='Generate resolved run/stage YAMLs without training.')
    multi.add_argument('--stop-timeout', type=int, default=120, help='Seconds allowed for graceful stopping.')
    multi.add_argument('--no-gui', action='store_true', help='Do not open the multi-training control window.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == 'multi':
        if args.stop_timeout <= 0:
            parser.error('--stop-timeout must be positive.')
        from library.multitraining import run_multi_training
        return run_multi_training(args)

    from library.configuration import load_config
    from library.task_check import check_config
    import yaml
    try:
        config = load_config(args.config, 'training')
        config.setdefault('model', {})['algorithm'] = str(config.get('model', {}).get('algorithm', '')).upper()
        report = check_config(config, 'training')
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    for warning in report['warnings']:
        print('[WARNING] ' + warning)
    if args.check:
        print('Task configuration: ' + ('PASS' if report['ok'] else 'NEEDS ATTENTION'))
        print(yaml.safe_dump(report['details'], sort_keys=False).strip())
        for error in report['errors']:
            print('ERROR: ' + error)
        print('No simulators started.')
        return 0 if report['ok'] else 1
    if not report['ok']:
        parser.error('\n'.join(report['errors']))

    from library.process_checks import require_idle, find_available_ports
    try:
        require_idle('Training')
        config['env']['base_port'] = find_available_ports(config['env']['base_port'],
                    config['env']['instances'] + 1 + config['train'].get('eval_instances', 1))
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    from library.training import run_training
    return run_training(config) or 0


if __name__ == '__main__':
    raise SystemExit(main())
