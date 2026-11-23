"""Plot completed benchmark trials recursively, including folders moved by hand."""
if __package__ in (None, ''):
    import _bootstrap

import argparse
import csv
import json
import math
from pathlib import Path


def read_trials(root):
    rows = []
    # Worker results are authoritative: do not also read their aggregate copies.
    for path in sorted(root.rglob('result.json')):
        try:
            row = json.loads(path.read_text())
            speed = float(row['samples_per_second'])
            count = int(row['instances'])
            if row.get('status') != 'ok' or not math.isfinite(speed) or speed <= 0 or count <= 0:
                continue
            rows.append(dict(instances=count, samples_per_second=speed,
                             repeat=row.get('repeat', 1), seed=row.get('seed'),
                             source=str(path)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print('Skipping %s: %s' % (path, exc))
    return sorted(rows, key=lambda r: (r['instances'], r['source']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path('logs/instance_benchmarks'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows = read_trials(args.directory)
    if not rows:
        parser.error('No successful result.json trials found in %s' % args.directory)
    output = args.output_dir or args.directory / 'combined_plots'
    output.mkdir(parents=True, exist_ok=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    speeds = [r['samples_per_second'] for r in rows]
    colors = ['#2563eb' if v == max(speeds) else '#dc2626' if v == min(speeds) else '#94a3b8' for v in speeds]
    bars = ax.bar(range(len(rows)), speeds, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(r['instances']) for r in rows])
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_x()+bar.get_width()/2, speed, '%.1f' % speed, ha='center', va='bottom')
    ax.set_ylim(0, max(speeds)*1.15)
    ax.set_xlabel('Parallel instances (one bar per completed trial)')
    ax.set_ylabel('Training speed (total steps/second)')
    ax.set_title('Saved benchmark trials — simulation, inference and learning')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.supxlabel('Blue: fastest trial. Red: slowest trial. Incomplete trials excluded; settings may differ across runs.')
    for extension in ('png', 'svg'):
        fig.savefig(output / ('speed_per_run.'+extension), dpi=160)
    with (output/'trials.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print('%3d instances: %.2f steps/s' % (row['instances'], row['samples_per_second']))
    print('Saved plot and source index to %s' % output.resolve())


if __name__ == '__main__':
    main()
