from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from hashlib import sha256

class StateCache(OrderedDict):

    def __init__(self, max_bytes=32 * 1024 ** 2):
        super().__init__()
        self.max_bytes = max_bytes
        self.bytes = 0

    @staticmethod
    def size(value):
        return value.shape[0] * value.shape[1] * 16

    def __setitem__(self, key, value):
        if key in self:
            self.bytes -= self.size(self[key])
        super().__setitem__(key, value)
        self.bytes += self.size(value)
        self.move_to_end(key)
        while self.bytes > self.max_bytes:
            self.popitem(last=False)

    def popitem(self, last=True):
        key, value = super().popitem(last=last)
        self.bytes -= self.size(value)
        return (key, value)

    def clear(self):
        super().clear()
        self.bytes = 0

class AcquisitionCache:

    def __init__(self):
        self.prefixes = StateCache()
        self.states = StateCache()
        self.metadata = {}
        self.identity = None
        self.stats = dict(state_hits=0, state_misses=0, prefix_hits=0, interval_solves=0, invalidations=0)

    def clear(self):
        self.prefixes.clear()
        self.states.clear()
        self.metadata.clear()

    def run(self, sim, sequence, rho0=None, t0=0.0):
        if 'flux_coupler' in sim.dev.physics:
            return sim.run(sequence, rho0=rho0, t0=t0, return_state=True)
        initial = b'' if rho0 is None else rho0.full().tobytes() + repr(rho0.dims).encode()
        identity = sha256(repr((type(sim), sim.dev.to_json(), sim.mode, sim.opts, sim.exact_idle, sim.exact_flat, sim.noise_order, sim.noise_max_order, sim.noise_tolerance, getattr(sim, 'hardware_controls', {}), t0)).encode() + initial).digest()
        if identity != self.identity:
            if self.identity is not None:
                self.stats['invalidations'] += 1
            self.clear()
            self.identity = identity
        key = sha256(repr(sequence).encode()).digest()
        if key in self.states:
            self.stats['state_hits'] += 1
            self.states.move_to_end(key)
            pops, end, noise = self.metadata[key]
            if noise is not None:
                sim.last_noise_convergence = deepcopy(noise)
            return (pops.copy(), self.states[key].copy(), end)
        self.stats['state_misses'] += 1
        from .hardware import dynamics_enabled
        if dynamics_enabled(sim.dev):
            if 'cosine_transmon' in sim.dev.physics:
                from .cosine_transmon import run
                result = run(sim, sequence, rho0, t0, True, _prefix_cache=self.prefixes, _stats=self.stats)
            else:
                from .extended_dynamics import run
                result = run(sim, sequence, rho0, t0, True, _prefix_cache=self.prefixes, _stats=self.stats, _strict_prefix=True)
        else:
            result = sim.run(sequence, rho0=rho0, t0=t0, return_state=True)
        pops, state, end = result
        self.states[key] = state.copy()
        self.metadata[key] = (pops.copy(), end, deepcopy(getattr(sim, 'last_noise_convergence', None)))
        self.metadata = {k: v for k, v in self.metadata.items() if k in self.states}
        return result

@contextmanager
def acquisition_cache(env):
    if getattr(env, '_acquisition_cache', None) is not None:
        yield env._acquisition_cache
        return
    cache = AcquisitionCache()
    env._acquisition_cache = cache
    try:
        yield cache
    finally:
        env.last_acquisition_cache_stats = cache.stats.copy()
        env._acquisition_cache = None
        cache.clear()

def reuse_acquisitions(function):

    @wraps(function)
    def wrapped(env, *args, **kwargs):
        with acquisition_cache(env):
            return function(env, *args, **kwargs)
    return wrapped
