from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from .sim_engine import Engine, sequence_to_wire, state_to_wire, state_from_wire, digest
_worker = None

def initialize(dev, options, barrier):
    global _worker
    _worker = Engine(dev, options)
    _worker.dressed()
    if _worker.D <= 81:
        _worker._build_idle_blocks()
    barrier.wait(timeout=120)

def ready():
    return True

def solve(job):
    return _worker.solve(job)

def solve_group(jobs):
    return [_worker.solve(job) for job in jobs]

class ParallelEngine(Engine):

    def __init__(self, dev, options, workers=2):
        super().__init__(dev, options)
        self.workers = workers
        context = get_context('spawn')
        barrier = context.Barrier(workers)
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize, initargs=(dev, options, barrier))
        startup = [self.pool.submit(ready) for _ in range(workers)]
        for future in startup:
            future.result(timeout=150)

    def run_many(self, sequences, rho0=None, t0=0.0, return_state=False, rho0s=None):
        sequences = list(sequences)
        initials = [rho0] * len(sequences) if rho0s is None else list(rho0s)
        if len(initials) != len(sequences):
            raise ValueError('rho0s must match sequences')
        from .three_qubit_runtime import supported
        if supported(self.dev):
            yield from self._three_run_many(sequences, initials, t0, return_state)
            return
        futures = [self.pool.submit(solve, dict(seq=sequence_to_wire(seq), rho0=state_to_wire(initial), t0=t0)) for seq, initial in zip(sequences, initials)]
        try:
            for future in futures:
                value = future.result()
                self.backend_stats = {**value['stats'], 'backend': 'parallel_full_lindblad'}
                pops = np.asarray(value['pops'])
                yield ((pops, state_from_wire(value['rho']), value['t']) if return_state else pops)
        finally:
            for future in futures:
                future.cancel()

    def _three_run_many(self, sequences, initials, t0, return_state):
        from .core import Segment
        groups = {}
        for index, (seq, initial) in enumerate(zip(sequences, initials)):
            job = dict(seq=sequence_to_wire(seq), rho0=state_to_wire(initial), t0=t0)
            key = ('point', index)
            for position, element in enumerate(seq):
                if isinstance(element, Segment) and element.pulses and self._is_flat_top(element):
                    prefix = sequence_to_wire(seq[:position + 1])
                    for p in prefix[-1]['pulses']:
                        p.pop('duration')
                    key = digest(dict(prefix=prefix, rho0=job['rho0'], t0=t0))
                    break
            groups.setdefault(key, []).append((index, job))
        pending = [(entries, self.pool.submit(solve_group, [job for _, job in entries])) for entries in groups.values()]
        try:
            ordered = [None] * len(sequences)
            for entries, future in pending:
                for (index, _), value in zip(entries, future.result()):
                    ordered[index] = value
            for value in ordered:
                self.backend_stats = {**value['stats'], 'backend': 'parallel_full_lindblad'}
                pops = np.asarray(value['pops'])
                yield ((pops, state_from_wire(value['rho']), value['t']) if return_state else pops)
        finally:
            for _, future in pending:
                future.cancel()

    def close(self):
        pool = getattr(self, 'pool', None)
        if pool is not None:
            self.pool = None
            pool.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        pool = getattr(self, 'pool', None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
