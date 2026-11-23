"""Opt-in, isolated PPO throughput benchmark. Default execution only plans jobs."""

if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.

import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import zipfile

import psutil
import yaml

from utils.benchmark_processes import exclusive_benchmark, run_worker
from library.configuration import load_config, DEFAULT_CONFIG
from library.common import resolve_path, resolve_model_path


from library.process_checks import active_simulators, require_idle, check_ports, find_available_ports


def plan(config, counts, requested_steps, repeats, seed, steps_per_instance=None,
         same_seed_repeats=False):
    if str(config['model']['algorithm']).upper() != 'PPO':
        raise ValueError('This benchmark currently supports PPO only.')
    counts = list(counts)
    def positive_integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if (not counts or not all(positive_integer(n) for n in counts)
            or not positive_integer(repeats) or not positive_integer(requested_steps)):
        raise ValueError('Counts, repeats and timesteps must be positive.')
    if steps_per_instance is not None and not positive_integer(steps_per_instance):
        raise ValueError('Steps per instance must be a positive integer.')
    counts = sorted(set(counts))
    n_steps = config['model']['n_steps']
    batch = config['model']['batch_size']
    if not positive_integer(n_steps) or not positive_integer(batch):
        raise ValueError('model.n_steps and model.batch_size must be positive integers.')
    if batch < 2 or min(counts) * n_steps < 2:
        raise ValueError('Invalid PPO rollout or minibatch size.')
    jobs = []
    # Round each count independently: the LCM of every count 10..40 is enormous.
    # Compare actual samples/second, not raw runtime of unequal rounded budgets.
    for count in counts:
        size = count * n_steps
        target = requested_steps if steps_per_instance is None else steps_per_instance * count
        budget = ((target + size - 1) // size) * size
        for repeat in range(repeats):
            jobs.append(dict(instances=count, repeat=repeat + 1,
                             seed=seed if same_seed_repeats else seed + repeat,
                             timesteps=budget, rollouts=budget // size,
                             rollout_size=size, batch_size=batch,
                             partial_minibatch=bool(size % batch)))
    return jobs


def checkpoint_path(value):
    """Accept the same extensionless model names as the repository/SB3 loader."""
    path = Path(resolve_model_path(value))
    if not path.is_file() and Path(str(path) + '.zip').is_file():
        path = Path(str(path) + '.zip')
    return str(path)


def validate_timing(config):
    for key in ('episode_duration', 'rl_observation_freq'):
        value = config['env'][key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('env.%s must be a positive finite number.' % key)


def prepare_config(config, count, port):
    result = deepcopy(config)
    result['env'].update(instances=count, base_port=port)
    result['sim'].update(graphical_interface=False, evaluation_graphical_interface=False)
    result.setdefault('log', {})['enable_wandb'] = False
    return result


def measured_learning(config, checkpoint, job, warmup_steps, device, factory=None):
    """Run one trial; factory injection permits real PPO tests without Stonefish."""
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

    class TimedPPO(PPO):
        update_seconds = 0.0
        update_calls = 0

        def train(self):
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            super().train()
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.update_seconds += time.perf_counter() - start
            self.update_calls += 1

    class Metrics(VecEnvWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.measuring = False
            self.latencies = []
            self.contact_latencies = []
            self.episodes = self.successes = self.proxy_samples = 0

        def reset(self):
            return self.venv.reset()

        def step_async(self, actions):
            self.started = time.perf_counter()
            self.venv.step_async(actions)

        def step_wait(self):
            obs, rewards, dones, infos = self.venv.step_wait()
            elapsed = time.perf_counter() - self.started
            if self.measuring:
                self.latencies.append(elapsed)
                proxy = [abs(float(i.get('acceleration', 0))) >= 3 for i in infos]
                self.proxy_samples += sum(proxy)
                if any(proxy):
                    self.contact_latencies.append(elapsed)
                for done, info in zip(dones, infos):
                    if done:
                        self.episodes += 1
                        self.successes += int(info.get('benchmark_success', False))
            return obs, rewards, dones, infos

    # These wrappers live only in benchmark workers; no production env changes.
    class SuccessInfo(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            info['benchmark_success'] = bool(info.get('is_success',
                terminated and getattr(self.env.unwrapped, 'goal_achieved', False)))
            return obs, reward, terminated, truncated, info

    def make(rank):
        def init():
            from library.env_factory import make_env_instance
            return SuccessInfo(make_env_instance(config['env'].get('type', 'docking'), rank, config))
        return init

    start = time.perf_counter()
    # Load on CPU/GPU before launching any simulator; reject invalid checkpoints early.
    m = config['model']
    lr = m.get('learning_rate', m.get('learning_rate_start', 3e-4))
    lr = 3e-4 if str(lr).lower() in ('auto', 'default') else float(lr)
    end_lr = m.get('learning_rate_end', lr)
    end_lr = 3e-4 if str(end_lr).lower() in ('auto', 'default') else float(end_lr)
    if 'learning_rate' in m:
        end_lr = lr
    schedule = lambda progress: end_lr + progress * (lr - end_lr)
    model = TimedPPO.load(checkpoint, device=device, n_steps=m['n_steps'],
                          n_envs=job['instances'],
                          batch_size=m['batch_size'], learning_rate=schedule,
                          clip_range=m.get('clip_range', 0.2), tensorboard_log=None,
                          seed=job['seed'])
    env = None
    try:
        env = Metrics(factory() if factory else SubprocVecEnv([make(i) for i in range(job['instances'])]))
        model.set_env(env)
        env.seed(job['seed'])
        obs = env.reset()
        for _ in range(warmup_steps):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        # Reset to the same seeded initial distribution after warm-up; no warm-up updates.
        model.set_random_seed(job['seed'])
        startup_seconds = time.perf_counter() - start
        env.measuring = True
        started = time.perf_counter()
        model.learn(total_timesteps=job['timesteps'], reset_num_timesteps=True)
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - started
        n = job['instances']
        steps = int(model.num_timesteps)
        latency = np.asarray(env.latencies)
        contacts = np.asarray(env.contact_latencies)
        return dict(job, status='ok', startup_warmup_seconds=startup_seconds,
                    learning_seconds=elapsed, actual_timesteps=steps,
                    samples_per_second=steps / elapsed,
                    per_instance_steps_per_second=steps / n / elapsed,
                    realtime_factor=steps / n / elapsed / config['env']['rl_observation_freq'],
                    ppo_update_seconds=model.update_seconds, ppo_update_calls=model.update_calls,
                    simulator_wait_seconds=float(latency.sum()),
                    step_p50_ms=float(np.percentile(latency, 50) * 1000),
                    step_p95_ms=float(np.percentile(latency, 95) * 1000),
                    step_p99_ms=float(np.percentile(latency, 99) * 1000),
                    step_max_ms=float(latency.max() * 1000),
                    collision_proxy_samples=env.proxy_samples,
                    collision_vector_steps=len(contacts),
                    collision_step_p95_ms=float(np.percentile(contacts, 95) * 1000) if len(contacts) else None,
                    episodes=env.episodes,
                    success_rate=env.successes / env.episodes if env.episodes else None,
                    device=str(model.device), n_epochs=model.n_epochs)
    finally:
        if env is not None:
            env.close()


def supervised_trial(payload, directory, timeout, threads):
    """Delegate lifecycle/ownership to a controller that survives worker failure."""
    payload_path = directory / 'job.json'
    payload_path.write_text(json.dumps(payload, indent=2))
    environment = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads))
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
    metrics = run_worker(
        [sys.executable, str(Path(__file__).resolve()), '--worker', str(payload_path)],
        directory / 'worker.log', environment, timeout)
    result = json.loads((directory / 'result.json').read_text())
    result.update(metrics)
    return result


def summarize(rows, repeats):
    groups = {}
    for row in rows:
        if row['status'] == 'ok':
            groups.setdefault(row['instances'], []).append(row)
    candidates = []
    for n, items in sorted(groups.items()):
        if len(items) != repeats:
            continue
        speeds = [r['samples_per_second'] for r in items]
        mean = statistics.mean(speeds)
        deviation = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
        candidates.append(dict(instances=n,
                               mean_samples_per_second=mean,
                               std_samples_per_second=deviation,
                               speed_cv=deviation / mean if mean > 0 else float('inf'),
                               median_samples_per_second=statistics.median(r['samples_per_second'] for r in items),
                               min_samples_per_second=min(r['samples_per_second'] for r in items),
                               max_samples_per_second=max(r['samples_per_second'] for r in items),
                               median_step_p95_ms=statistics.median(r['step_p95_ms'] for r in items),
                               collision_proxy_samples=sum(r['collision_proxy_samples'] for r in items)))
    if not candidates:
        return dict(recommendation=None, candidates=[], repeats=repeats)
    peak = max(c['mean_samples_per_second'] for c in candidates)
    near_best = [c for c in candidates if c['mean_samples_per_second'] >= peak * .95]
    selected = min(near_best, key=lambda c: c['instances'])
    return dict(recommendation=selected['instances'], candidates=candidates, repeats=repeats,
                criterion='Smallest count within 5% of peak mean end-to-end training throughput.',
                collision_coverage=all(c['collision_proxy_samples'] > 0 for c in candidates))


def stopping_reason(summary, plateau_cv=0.01):
    """Check the last five completed counts, using their mean trial speeds."""
    tail = summary['candidates'][-5:]
    if len(tail) < 5:
        return None
    speeds = [candidate['mean_samples_per_second'] for candidate in tail]
    # Give a consistent decline precedence over a low-variation plateau.
    if all(right < left for left, right in zip(speeds, speeds[1:])):
        return 'decreasing_speed: last five tested counts each slower than the previous'
    mean = statistics.mean(speeds)
    variation = statistics.stdev(speeds) / mean if mean > 0 else float('inf')
    if variation < plateau_cv:
        return ('constant_speed: last five tested counts have speed CV %.3f%% below %.3f%%' %
                (100 * variation, 100 * plateau_cv))
    return None


def plot_results(summary, directory):
    """Standalone headless figure; no simulated or extrapolated data."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch
    candidates = summary['candidates']
    figure = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    if candidates:
        means = [c['mean_samples_per_second'] for c in candidates]
        best, worst = max(means), min(means)
        colors = ['#2563eb' if value == best else '#dc2626' if value == worst
                  else '#94a3b8' for value in means]
        uncertainty = {} if summary.get('repeats') == 1 else dict(
            yerr=[c['std_samples_per_second'] for c in candidates], capsize=3)
        ax.bar([c['instances'] for c in candidates], means, color=colors, **uncertainty)
        ax.set_xticks([c['instances'] for c in candidates])
        legend = [Patch(color='#2563eb', label='Fastest measured mean')]
        if best != worst:
            legend.append(Patch(color='#dc2626', label='Slowest measured mean'))
        ax.legend(handles=legend)
    else:
        ax.text(.5, .5, 'No completed instance counts yet', ha='center', transform=ax.transAxes)
    ax.set_xlabel('Parallel simulator instances')
    ax.set_ylabel('Average training speed (total samples/second)')
    ax.set_title('PPO training throughput — inference, simulation and learning included')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    reason = summary.get('stop_reason') or 'Sweep in progress'
    note = ('One trial per count; repeat variability not measured.  '
            if summary.get('repeats') == 1 else
            'Error bars: ±1 standard deviation across repeats.  ')
    figure.supxlabel(note + reason, fontsize=9)
    for suffix in ('png', 'svg'):
        figure.savefig(str(Path(directory) / ('speed_vs_instances.' + suffix)), dpi=160)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Training YAML matching the checkpoint; flat generated YAML is supported.')
    parser.add_argument('--model', help='Successful PPO checkpoint (.zip); required even for planning.')
    parser.add_argument('--instances', nargs='+', type=int, default=list(range(10, 41)), help='Ascending counts; default: every integer from 10 through 40.')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--timesteps', type=int, default=240000, help='Minimum samples per trial; rounded up independently to complete PPO rollouts.')
    budget.add_argument('--steps-per-instance', type=int, help='Minimum steps per worker per trial; rounds up to a multiple of model.n_steps. Overrides the default total budget.')
    parser.add_argument('--repeats', type=int, default=3, help='Trials per instance count; minimum 1 (default: 3).')
    parser.add_argument('--plateau-cv', type=float, default=0.01, help='Stop when std/mean of the last five counts mean speeds is below this fraction (default 0.01).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--same-seed-repeats', action='store_true',
                        help='Reuse the base seed for every repeat as well as every instance count.')
    parser.add_argument('--base-port', type=int, default=6500)
    parser.add_argument('--warmup-steps', type=int, help='Default: one full episode per instance, without learning.')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--timeout', type=float, default=3600, help='Maximum seconds per trial including initialization.')
    parser.add_argument('--output', default='logs/instance_benchmarks')
    parser.add_argument('--run', action='store_true', help='Launch real benchmark, only when Stonefish is idle.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))
        payload = json.loads(Path(args.worker).read_text())
        result = measured_learning(**payload)
        Path(args.worker).with_name('result.json').write_text(json.dumps(result, indent=2))
        return
    if not args.model:
        parser.error('--model is required; choose a successful PPO checkpoint and its matching YAML.')
    if (args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or (args.warmup_steps is not None and args.warmup_steps < 0)):
        parser.error('Invalid thread count, timeout or warm-up length.')
    if args.repeats < 1 or not math.isfinite(args.plateau_cv) or args.plateau_cv <= 0:
        parser.error('Use at least one repeat and a positive finite plateau CV threshold.')
    config = load_config(args.config, 'training')
    validate_timing(config)
    if args.seed < 0 or args.seed + args.repeats + max(args.instances) - 2 >= 2**32:
        parser.error('Seed plus repeat/worker offsets must fit an unsigned 32-bit integer.')
    checkpoint = checkpoint_path(args.model)
    with zipfile.ZipFile(checkpoint) as archive:
        metadata = json.loads(archive.read('data'))
        if 'clip_range' not in metadata or 'n_epochs' not in metadata:
            raise ValueError('Expected a PPO checkpoint.')
    jobs = plan(config, args.instances, args.timesteps, args.repeats, args.seed,
                steps_per_instance=args.steps_per_instance,
                same_seed_repeats=args.same_seed_repeats)
    if args.base_port < 1024 or args.base_port + max(args.instances) > 65536:
        parser.error('Invalid simulator port range.')
    warmup = args.warmup_steps if args.warmup_steps is not None else math.ceil(
        config['env']['episode_duration'] * config['env']['rl_observation_freq'])
    print(json.dumps(dict(checkpoint=checkpoint, config=args.config, warmup_steps=warmup,
                          mode='RUN' if args.run else 'PLAN ONLY (no simulators or learning)',
                          active_simulators=active_simulators(), jobs=jobs), indent=2))
    if not args.run:
        return
    with exclusive_benchmark():
        run_sweep(args, config, checkpoint, jobs, warmup)


def run_sweep(args, config, checkpoint, jobs, warmup):
    """Run under the sweep lock; persist every completed/failed/interrupted trial."""
    require_idle()
    import datetime
    directory = Path(resolve_path(args.output)) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'plan.json').write_text(json.dumps(vars(args), indent=2))
    (directory / 'system.json').write_text(json.dumps(dict(cpu_count=psutil.cpu_count(),
        physical_cores=psutil.cpu_count(logical=False), ram_bytes=psutil.virtual_memory().total,
        python=sys.version, platform=sys.platform), indent=2))
    rows = []
    for index, job in enumerate(jobs):
        require_idle()
        trial_port = find_available_ports(args.base_port, job['instances'])
        trial_dir = directory / ('%02d_n%d_repeat%d' % (index + 1, job['instances'], job['repeat']))
        trial_dir.mkdir()
        cfg = prepare_config(config, job['instances'], trial_port)
        (trial_dir / 'training.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        payload = dict(config=cfg, checkpoint=checkpoint, job=job, warmup_steps=warmup, device=args.device)
        print('Trial %d/%d: %s' % (index + 1, len(jobs), job), flush=True)
        started = time.perf_counter()
        try:
            result = supervised_trial(payload, trial_dir, args.timeout, args.threads)
        except KeyboardInterrupt:
            result = dict(job, status='interrupted', error='Benchmark interrupted by user or SIGTERM.')
        except Exception as exc:
            result = dict(job, status='failed', error=str(exc))
        result['trial_wall_seconds'] = time.perf_counter() - started
        rows.append(result)
        (directory / 'results.json').write_text(json.dumps(rows, indent=2))
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / 'results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows, args.repeats)
        reason = ('interrupted' if result['status'] == 'interrupted' else
                  'trial_failed' if result['status'] != 'ok' else None)
        if not reason and job['repeat'] == args.repeats:
            reason = stopping_reason(summary, args.plateau_cv)
        if not reason and index == len(jobs) - 1:
            reason = 'completed_requested_instances'
        summary['stop_reason'] = reason
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        if job['repeat'] == args.repeats or reason:
            plot_results(summary, directory)
        if reason:
            break
    print('Results: %s\n%s' % (directory, json.dumps(summary, indent=2)))
    if any(row['status'] == 'interrupted' for row in rows):
        raise SystemExit(130)
    if any(row['status'] != 'ok' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
