"""Read-only simulator and port availability checks."""
import socket
import errno
import psutil

def active_simulators():
    result = []
    for proc in psutil.process_iter(['pid', 'name']):
        if (proc.info['name'] or '').startswith('StonefishRL'):
            result.append(proc.info['pid'])
    return result


def require_idle(workflow='Benchmark'):
    active = active_simulators()
    if active:
        raise RuntimeError('%s refused: Stonefish is already running (PIDs %s). '
                           'Wait until existing training/testing has finished.' % (workflow, active))


def check_ports(base, count):
    held = []
    try:
        for port in range(base, base + count):
            sock = socket.socket()
            held.append(sock)
            sock.bind(('0.0.0.0', port))
    finally:
        for sock in held:
            sock.close()


def find_available_ports(base, count):
    """Find a contiguous range, skipping busy ports (including TCP TIME_WAIT).

    This is a preflight check, not a reservation: sockets close before launch.
    Never terminate a process or enable port sharing to obtain the range.
    """
    if type(base) is not int or type(count) is not int or base < 1024 or count < 1:
        raise ValueError('Invalid simulator port range.')
    for candidate in range(base, 65537 - count):
        try:
            check_ports(candidate, count)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
        else:
            if candidate != base:
                print(f'[INFO] Ports starting at {base} unavailable; using {candidate}..{candidate + count - 1}.')
            return candidate
    raise RuntimeError(f'No free contiguous range of {count} ports at or above {base}.')


