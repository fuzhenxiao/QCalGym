from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import numpy as np
import qutip
import scipy
from .sim_engine import sequence_to_wire, state_to_wire, state_from_wire, solve_job

def backend_mode():
    mode = os.environ.get('QCAL_SIM_BACKEND', 'off')
    if mode not in ('off', 'local', 'service'):
        raise ValueError('QCAL_SIM_BACKEND must be off, local, or service')
    return mode

def enabled(sim):
    from .hardware import dynamics_enabled
    return 'flux_coupler' not in sim.dev.physics and (not dynamics_enabled(sim.dev)) and (sim.mode == 'qutip') and (not getattr(sim, '_backend_worker', False)) and (backend_mode() != 'off')

@lru_cache(maxsize=1)
def version():
    h = hashlib.sha256()
    base = Path(__file__).parent
    for name in ('core.py', 'device.py', 'sim_engine.py', 'sim_client.py', 'sim_service.py'):
        h.update((base / name).read_bytes())
    h.update(f'{sys.version}|{qutip.__version__}|{scipy.__version__}|{np.__version__}'.encode())
    return h.hexdigest()[:20]

def socket_path():
    override = os.environ.get('QCAL_SIM_SOCKET')
    if override:
        return Path(override)
    return Path('/tmp') / f'qcal-sim-{os.getuid()}' / f'{version()}.sock'

def control(path=None, op='status'):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        sock.connect(str(path or socket_path()))
        sock.sendall(json.dumps({'op': op}).encode() + b'\n')
        with sock.makefile('rb') as stream:
            return json.loads(stream.readline())

def ensure_service():
    path = socket_path()
    try:
        control(path)
        return path
    except (OSError, ValueError):
        pass
    path.parent.mkdir(mode=448, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid() or path.parent.stat().st_mode & 63:
        raise PermissionError('Simulation socket directory must be private to the current user')
    with path.with_suffix('.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            control(path)
            return path
        except (OSError, ValueError):
            pass
        workers = int(os.environ.get('QCAL_SIM_WORKERS', min(4, os.cpu_count() or 1)))
        if not 1 <= workers <= 64:
            raise ValueError('QCAL_SIM_WORKERS must be between 1 and 64')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', QCAL_SIM_BACKEND='off', MPLBACKEND='Agg')
        env.pop('QCAL_RUN_CONFIG', None)
        for key in list(env):
            if key.endswith(('_API_KEY', '_API_TOKEN')):
                env.pop(key)
        package_root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = package_root
        env.setdefault('MPLCONFIGDIR', str(path.parent / 'matplotlib'))
        log_path = path.with_suffix('.log')
        with log_path.open('ab') as log:
            process = subprocess.Popen([sys.executable, '-m', 'backend.sim_service', '--socket', str(path), '--workers', str(workers)], cwd=package_root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        threading.Thread(target=process.wait, daemon=True).start()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                control(path)
                return path
            except (OSError, ValueError):
                if process.poll() is not None:
                    raise RuntimeError(f'Simulation service failed to start; see {log_path}')
                time.sleep(0.1)
        raise TimeoutError(f'Simulation service startup timed out; see {log_path}')

def solve_many(sim, sequences, rho0=None, t0=0.0, return_state=False, rho0s=None):
    sequences = list(sequences)
    if not sequences:
        return
    initials = [rho0] * len(sequences) if rho0s is None else list(rho0s)
    if len(initials) != len(sequences):
        raise ValueError('rho0s must match sequences')
    spec = {'device': json.loads(sim.dev.to_json()), 'options': {**sim.opts, 'exact_flat': sim.exact_flat, 'exact_idle': sim.exact_idle}}
    jobs = [dict(seq=sequence_to_wire(seq), rho0=state_to_wire(initial), t0=float(t0)) for seq, initial in zip(sequences, initials)]

    def decode(value):
        sim.backend_stats = dict(value['stats'], backend=backend_mode(), version=version())
        pops = np.asarray(value['pops'])
        return (pops, state_from_wire(value['rho']), value['t']) if return_state else pops
    if backend_mode() == 'local':
        for job in jobs:
            yield decode(solve_job(spec, job))
        return
    path = ensure_service()
    for start in range(0, len(jobs), 441):
        batch = jobs[start:start + 441]
        message = dict(op='solve', request_id=uuid.uuid4().hex, spec=spec, jobs=batch)
        payload = json.dumps(message, separators=(',', ':')).encode() + b'\n'
        if len(payload) > 64 * 1024 ** 2:
            raise ValueError('Simulation batch exceeds 64 MiB; use smaller batches')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(str(path))
            sock.sendall(payload)
            with sock.makefile('rb') as stream:
                buffered = {}
                next_index = 0
                while next_index < len(batch):
                    line = stream.readline(16 * 1024 ** 2)
                    if not line:
                        raise ConnectionError('Simulation service disconnected; no automatic local fallback')
                    response = json.loads(line)
                    if 'index' not in response:
                        raise RuntimeError(response.get('error', 'Invalid simulation response'))
                    buffered[response['index']] = response
                    while next_index in buffered:
                        item = buffered.pop(next_index)
                        if 'error' in item:
                            raise RuntimeError(f"Simulation point {start + next_index}: {item['error']}")
                        yield decode(item['value'])
                        next_index += 1
