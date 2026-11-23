"""Own and clean up benchmark workers without signaling unrelated processes."""
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

import psutil


_TOKEN_KEY = 'STONEFISH_INSTANCE_BENCHMARK_TOKEN'


@contextmanager
def exclusive_benchmark(lock_path=None):
    """Prevent two sweeps by this user in the same host namespace from overlapping."""
    path = Path(lock_path or '/tmp/stonefish-instance-benchmark-%d.lock' % os.getuid())
    # Keep the inode after release: unlinking permits a third process to lock a
    # different inode while a second process still holds the original one.
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(str(path), flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another instance benchmark is already running for this user.') from None
        os.ftruncate(descriptor, 0)
        os.write(descriptor, ('%d\n' % os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _interrupt_on_sigterm():
    """Turn normal termination into an unwind so our ownership cleanup runs."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt('Benchmark interrupted by SIGTERM')

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _discover_owned(token, owned):
    """Environment marker survives reparenting and start_new_session=True."""
    for process in psutil.process_iter(['pid', 'uids']):
        try:
            if process.info['uids'].effective != os.geteuid():
                continue
            if process.environ().get(_TOKEN_KEY) == token:
                # Use the psutil Process itself, not a bare PID: signal methods
                # check its recorded creation time to protect against PID reuse.
                process.create_time()
                owned[(process.pid, process.create_time())] = process
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue


def _alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _signal_owned(processes, force=False):
    for process in processes:
        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except psutil.NoSuchProcess:
            pass


def _cleanup_owned(token, owned, worker):
    # Include a final token scan even if the worker exited between two polls.
    # A detached simulator may now have PID 1 as parent and is no longer in the
    # worker's psutil descendants, but it still carries this unique token.
    _discover_owned(token, owned)
    _signal_owned([process for process in owned.values() if _alive(process)])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        _discover_owned(token, owned)
        alive = [process for process in owned.values() if _alive(process)]
        if not alive:
            break
        # Newly discovered descendants must also receive termination.
        _signal_owned(alive)
        time.sleep(0.05)
    # Terminating a process that was still launching descendants can expose more
    # processes on the final scan. Kill those too, then verify shutdown.
    _discover_owned(token, owned)
    _signal_owned([process for process in owned.values() if _alive(process)], force=True)
    if worker is not None:
        worker.wait(timeout=5)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        _discover_owned(token, owned)
        alive = [process for process in owned.values() if _alive(process)]
        if not alive:
            return
        _signal_owned(alive, force=True)
        time.sleep(0.05)
    raise RuntimeError('Some benchmark-owned processes did not exit: %s' %
                       [process.pid for process in alive])


def run_worker(command, log_path, environment, timeout):
    """Run one worker and clean its entire token-owned tree on every exit path.

    Returns sampled memory metrics on success. Nonzero exit and timeout raise
    after cleanup. The unique environment marker is injected into a copy of
    ``environment``; it is never installed into the controller's environment.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Worker timeout must be positive and finite.')
    token = uuid.uuid4().hex
    worker_environment = dict(environment, **{_TOKEN_KEY: token})
    owned = {}
    worker = None
    peak_rss = 0
    min_available = psutil.virtual_memory().available
    with _interrupt_on_sigterm(), Path(log_path).open('w') as log:
        try:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      env=worker_environment, start_new_session=True)
            try:
                process = psutil.Process(worker.pid)
                owned[(process.pid, process.create_time())] = process
            except psutil.NoSuchProcess:
                pass  # A fast exit still gets the final orphan-token scan.
            started = time.monotonic()
            while worker.poll() is None:
                _discover_owned(token, owned)
                rss = 0
                for process in owned.values():
                    try:
                        rss += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_rss = max(peak_rss, rss)
                min_available = min(min_available, psutil.virtual_memory().available)
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('Trial exceeded %.1f seconds; see %s' % (timeout, log_path))
                time.sleep(0.1)
            if worker.returncode != 0:
                raise RuntimeError('Worker exited with code %d; see %s' % (worker.returncode, log_path))
            return dict(peak_process_tree_rss_bytes=peak_rss,
                        min_system_available_memory_bytes=min_available)
        finally:
            _cleanup_owned(token, owned, worker)
