"""Wait for a specific benchmark and apply its recommendation to training.env."""

if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.

import argparse
import json
from pathlib import Path
import os
import tempfile

import psutil
import yaml


def instance_node(text):
    node = yaml.compose(text)
    for key in ('training', 'env', 'instances'):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError('Expected training.env.instances in the YAML.')
        node = next((value for name, value in node.value if name.value == key), None)
    if not isinstance(node, yaml.ScalarNode) or node.tag != 'tag:yaml.org,2002:int':
        raise ValueError('training.env.instances must be an explicit integer.')
    return node


def apply_result(config_path, summary, expected_count, result_dir):
    reason = summary.get('stop_reason') or ''
    if not (reason == 'completed_requested_instances' or
            reason.startswith(('constant_speed:', 'decreasing_speed:'))):
        raise ValueError('Benchmark did not finish successfully; YAML left unchanged.')
    count = summary.get('recommendation')
    candidates = [c['instances'] for c in summary.get('candidates', [])]
    if type(count) is not int or count < 1 or count not in candidates:
        raise ValueError('No valid completed recommendation; YAML left unchanged.')
    original = config_path.read_text()
    node = instance_node(original)
    if int(node.value) != expected_count:
        raise ValueError('Instance count was edited while waiting; YAML left unchanged.')
    updated = original[:node.start_mark.index] + str(count) + original[node.end_mark.index:]
    assert yaml.safe_load(updated)['training']['env']['instances'] == count
    backup = result_dir / 'training_config_before_instance_update.yaml'
    with backup.open('x') as stream:
        stream.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=config_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
        temporary.chmod(config_path.stat().st_mode & 0o777)
        if config_path.read_text() != original:
            raise RuntimeError('Config changed during update; YAML left unchanged.')
        os.replace(str(temporary), str(config_path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    record = dict(config=str(config_path), previous_instances=expected_count,
                  instances=count, criterion=summary.get('criterion'), stop_reason=reason)
    (result_dir / 'config_update.json').write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    expected_count = int(instance_node(args.config.read_text()).value)
    process = psutil.Process(args.pid)
    if not any(Path(arg).name == 'benchmark_instances.py' or arg == 'utils.benchmark_instances'
               for arg in process.cmdline()):
        raise ValueError('PID is not a benchmark supervisor.')
    process.create_time()  # Bind this process identity; do not follow a reused PID.
    print('Waiting for benchmark PID %d; current training instances: %d' %
          (args.pid, expected_count), flush=True)
    while True:
        try:
            process.wait(timeout=15)
            break
        except psutil.TimeoutExpired:
            continue
    summary = json.loads((args.results / 'summary.json').read_text())
    print(json.dumps(apply_result(args.config, summary, expected_count, args.results), indent=2), flush=True)


if __name__ == '__main__':
    main()
