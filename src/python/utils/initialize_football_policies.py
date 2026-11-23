"""Create ten independent policy files from a trained team-football solo PPO."""
if __package__ in (None, ''):
    import _bootstrap
import argparse
import json
import shutil
from pathlib import Path
from library.team_football import AGENTS, ROLES, OBSERVATION_SIZE, ACTION_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing directories are never overwritten.')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new directory.')
    from stable_baselines3 import PPO
    model = PPO.load(str(args.model), device='cpu')
    if model.observation_space.shape != (OBSERVATION_SIZE,) or model.action_space.shape != (ACTION_SIZE,):
        parser.error('Expected a team-football policy with 62 observations and 9 actions.')
    source = args.model if args.model.is_file() else Path(str(args.model)+'.zip')
    args.output_dir.mkdir(parents=True)
    roster = {}
    for agent in AGENTS:
        output = args.output_dir / (agent+'.zip')
        shutil.copy2(source, output)
        roster[agent] = dict(policy=output.name, role=ROLES[agent], team=agent.split('_')[0])
    (args.output_dir/'roster.json').write_text(json.dumps(dict(
        source=str(source.resolve()), agents=roster,
        status='Initialized only; shared-match training and robot variants are not implemented yet.'), indent=2))
    print('Created ten independent PPO archives and roster.json in %s' % args.output_dir)


if __name__ == '__main__':
    main()
