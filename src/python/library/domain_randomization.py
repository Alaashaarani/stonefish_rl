"""Seeded episode parameters, resolved from source XML for runtime RESET updates."""
from copy import deepcopy
from pathlib import Path
import math
import subprocess
import tempfile
import yaml
from library.common import resolve_path


def expand_targets(document):
    """Expand shared group bounds to individual targets for the existing C++ API."""
    if not isinstance(document, dict):
        raise ValueError('Domain randomization YAML must be a mapping.')
    targets = deepcopy(document.get('targets', []))
    if not isinstance(targets, list):
        raise ValueError('Domain randomization targets must be a list.')
    groups = document.get('groups', {})
    if not isinstance(groups, dict):
        raise ValueError('Domain randomization groups must be a mapping.')
    for group_id, group in groups.items():
        if not isinstance(group_id, str) or not group_id or not isinstance(group, dict):
            raise ValueError('Each domain randomization group needs a name and mapping.')
        if set(group) - {'file', 'kind', 'parts', 'parameters'}:
            raise ValueError('Unknown fields in domain randomization group: ' + group_id)
        for key in ('file', 'kind'):
            if not isinstance(group.get(key), str) or not group[key]:
                raise ValueError(f'Group {group_id} requires {key}.')
        parts, parameters = group.get('parts'), group.get('parameters')
        if not isinstance(parts, list) or not parts:
            raise ValueError(f'Group {group_id} requires a nonempty parts list.')
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f'Group {group_id} requires a nonempty parameters mapping.')
        member_ids = set()
        for part in parts:
            if not isinstance(part, dict) or set(part) - {'id', 'name', 'occurrence'}:
                raise ValueError(f'Invalid part fields in group {group_id}.')
            member_id = part.get('id')
            if not isinstance(member_id, str) or not member_id or member_id in member_ids:
                raise ValueError(f'Group {group_id} requires unique nonempty part IDs.')
            member_ids.add(member_id)
            if not isinstance(part.get('name'), str) or not part['name']:
                raise ValueError(f'Part {member_id} requires an XML name.')
            if 'occurrence' in part and (type(part['occurrence']) is not int or part['occurrence'] < 0):
                raise ValueError(f'Part {member_id} occurrence must be a nonnegative integer.')
            for parameter, bounds in parameters.items():
                if not isinstance(parameter, str) or not isinstance(bounds, dict) or set(bounds) != {'min', 'max'}:
                    raise ValueError(f'Group {group_id} parameter {parameter} requires min and max only.')
                target = {'id': f'{group_id}.{member_id}.{parameter}',
                          'file': group['file'], 'kind': group['kind'], 'name': part['name'],
                          'parameter': parameter, **deepcopy(bounds)}
                if 'occurrence' in part:
                    target['occurrence'] = part['occurrence']
                targets.append(target)
    return targets


class EpisodeRandomizer:
    def __init__(self, config):
        self.config = deepcopy(config)
        settings = config.get('domain_randomization', {})
        self.mode = settings.get('mode', 'runtime')
        if self.mode not in ('runtime', 'restart'):
            raise ValueError('library.domain_randomization.mode must be runtime or restart')
        self.editor = Path(resolve_path(settings.get('editor', 'build/StonefishDomainRandomization')))
        if not self.editor.is_file():
            raise ValueError('Build StonefishDomainRandomization before enabling domain randomization.')
        self.target_path = resolve_path(settings['config'])
        self._load_targets()
        self.workspace = tempfile.TemporaryDirectory(prefix='stonefish-domain-')
        self.episode = 0
        self.applied = []

    def _load_targets(self):
        # Saved bounds/target changes take effect on the next episode reset.
        with open(self.target_path) as stream:
            document = yaml.safe_load(stream)
        self.targets = expand_targets(document)
        if not isinstance(self.targets, list) or (not self.targets and self.mode == 'restart'):
            raise ValueError('Domain randomization targets must be a list (nonempty in restart mode).')
        ids = set()
        for target in self.targets:
            if target['id'] in ids:
                raise ValueError('Duplicate domain randomization id: ' + target['id'])
            ids.add(target['id'])
            size = {'mass': 1, 'max_setpoint': 1, 'compound_transform.xyz': 3,
                    'compound_transform.rpy': 3}.get(target['parameter'])
            if size is None:
                raise ValueError('Unsupported parameter: ' + target['parameter'])
            low, high = self._bounds(target['min'], size), self._bounds(target['max'], size)
            if any(a > b for a, b in zip(low, high)):
                raise ValueError('Domain randomization min exceeds max: ' + target['id'])

    @staticmethod
    def _bounds(value, size):
        values = [value] if size == 1 else value
        if not isinstance(values, list) or len(values) != size:
            raise ValueError('Wrong domain randomization bounds shape.')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError('Domain randomization bounds must be finite numbers.')
        return values

    def prepare(self, rng):
        self._load_targets()
        targets = deepcopy(self.targets)
        for target in targets:
            size = 1 if target['parameter'] in ('mass', 'max_setpoint') else 3
            low, high = self._bounds(target['min'], size), self._bounds(target['max'], size)
            delta = [float(rng.uniform(a, b)) for a, b in zip(low, high)]
            target['delta'] = delta[0] if size == 1 else delta
        root = Path(self.workspace.name)
        payload, report_path = root / 'parameters.yaml', root / 'applied.yaml'
        payload.write_text(yaml.safe_dump({'targets': targets}))
        sim = self.config['sim']
        command = [str(self.editor), resolve_path(sim['scene_path']),
                   resolve_path(sim['resources_path']), str(payload), str(root / 'scene'), str(report_path)]
        if self.mode == 'runtime':
            command.append('--report-only')
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError('Domain randomization rejected: ' + result.stderr.strip())
        report = yaml.safe_load(report_path.read_text())
        self.applied = report['applied']
        self.episode += 1
        config = deepcopy(self.config)
        if self.mode == 'restart':
            config['sim']['scene_path'] = report['scene_path']
        elif any('runtime' not in target for target in self.applied):
            raise ValueError('Runtime randomization requires robot targets resolved by the current XML editor')
        return config

    def close(self):
        self.workspace.cleanup()


def configure_domain_randomization(config):
    settings = config.get('domain_randomization', {})
    return EpisodeRandomizer(config) if settings.get('enabled', False) else None
