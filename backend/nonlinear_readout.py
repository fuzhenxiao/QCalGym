from copy import deepcopy
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

def enable_nonlinear_readout(dev, *, g=0.1, levels=6, drive_scale=0.001, temporal=False):
    out = deepcopy(dev)
    out.physics['readout'] = {**out.physics.get('readout', {}), 'temporal': temporal, 'nonlinear': [dict(g=g, levels=levels, drive_scale=drive_scale) for _ in out.qubits]}
    validate_readout(out)
    return out

def validate_readout(dev):
    from .instruments import validate
    validate(dev)
    ro = dev.physics.get('readout', {})
    if not isinstance(ro, dict) or set(ro) - {'temporal', 'bins', 'nonlinear', 'electronics'}:
        raise ValueError('Unknown readout model parameter')
    if type(ro.get('temporal', False)) is not bool:
        raise ValueError('temporal must be boolean')
    if type(ro.get('bins', 64)) is not int or not 8 <= ro.get('bins', 64) <= 256:
        raise ValueError('Readout bins must be an integer in [8,256]')
    entries = ro.get('nonlinear', [None] * dev.n)
    if not isinstance(entries, list) or len(entries) != dev.n:
        raise ValueError('nonlinear requires one entry per readout resonator')
    for k, p in enumerate(entries):
        if p is None:
            continue
        if not isinstance(p, dict) or set(p) != {'g', 'levels', 'drive_scale'}:
            raise ValueError('Nonlinear readout requires g, levels, drive_scale')
        if type(p['levels']) is not int or not max(3, dev.qubits[k].n_levels) <= p['levels'] <= 10:
            raise ValueError('Readout dressing levels must be an integer in [3,10]')
        for name, lo, hi in [('g', 0.0, 0.5), ('drive_scale', 1e-07, 0.1)]:
            v = p[name]
            if isinstance(v, bool) or not np.isscalar(v) or (not np.isfinite(v)) or (not lo <= v <= hi):
                raise ValueError(f'Invalid nonlinear readout {name}')
        if not np.isfinite(dev.resonators[k].kappa) or dev.resonators[k].kappa <= 0:
            raise ValueError('Readout linewidth must be positive')
        transitions = dev.qubits[k].freq + dev.qubits[k].anharm * np.arange(p['levels'] - 1)
        if np.min(transitions) <= 0 or np.min(abs(transitions - dev.resonators[k].freq)) < 2 * p['g']:
            raise ValueError('Dressing model needs positive, dispersively detuned local transitions')

def validate_amplitudes(dev, values):
    for k, v in values.items():
        if type(k) is not int or not 0 <= k < dev.n or isinstance(v, bool) or (not np.isscalar(v)) or (not np.isfinite(v)) or (not 0 <= v <= 1000):
            raise ValueError('readout_amplitude requires valid qubit indices and finite values in [0,1000]')

class DressedCavity:

    def __init__(self, fq, anharm, fr, g, levels):
        self.fr = fr
        self.photons = np.unique(np.r_[np.arange(129), np.round(np.geomspace(129, 10000000000.0, 1800))])
        self.frequencies = np.empty((3, len(self.photons)))

        def block(total):
            j = np.arange(min(levels, int(total) + 1))
            diag = j * (fq - fr) + anharm * j * (j - 1) / 2
            return eigh_tridiagonal(diag, g * np.sqrt(j[1:] * (total - j[1:] + 1)))
        low = self.photons < levels - 1
        for col in np.flatnonzero(low):
            photons = self.photons[col]
            blocks = [block(photons + i) for i in range(4)]
            for state in range(3):
                e, v = blocks[state]
                j = np.arange(len(e))
                bare = j * (fq - fr) + anharm * j * (j - 1) / 2
                index = int(np.flatnonzero(np.argsort(bare) == state)[0])
                ep, vp = blocks[state + 1]
                vec = np.pad(v[:, index], (0, len(ep) - len(e)))
                successor = np.argmax(abs(vec @ vp) ** 2)
                self.frequencies[state, col] = fr + ep[successor] - e[index]
        n = self.photons[~low]
        totals = np.unique((n[:, None] + np.arange(4)).ravel())
        energies = np.empty((len(totals), levels))
        vectors = np.empty((len(totals), levels, levels))
        j = np.arange(levels)
        diagonal = j * (fq - fr) + anharm * j * (j - 1) / 2
        ranks = np.argsort(np.argsort(diagonal))
        for start in range(0, len(totals), 256):
            ns = totals[start:start + 256]
            h = np.zeros((len(ns), levels, levels))
            h[:, j, j] = diagonal
            coupling = g * np.sqrt(j[1:] * (ns[:, None] - j[1:] + 1))
            h[:, j[:-1], j[1:]] = coupling
            h[:, j[1:], j[:-1]] = coupling
            energies[start:start + len(ns)], vectors[start:start + len(ns)] = np.linalg.eigh(h)
        for state in range(3):
            ids = np.searchsorted(totals, n + state)
            rank = ranks[state]
            overlap = np.einsum('bi,bij->bj', vectors[ids, :, rank], vectors[ids + 1], optimize=False)
            successor = np.argmax(abs(overlap) ** 2, axis=1)
            self.frequencies[state, ~low] = fr + energies[ids + 1, successor] - energies[ids, rank]

    def frequency(self, photons, state):
        n = np.asarray(photons)
        if np.any(n < 0) or np.any(n > self.photons[-1]) or (not np.isfinite(n).all()):
            raise ValueError('Cavity photons outside cached model domain [0,1e10]')
        if np.ndim(state) == 0:
            return np.interp(n, self.photons, self.frequencies[int(state)])
        return np.choose(state, [np.interp(n, self.photons, row) for row in self.frequencies])

    def steady(self, frequency, kappa, epsilon, state):
        if epsilon == 0:
            return (0.0, float(self.frequency(0, state)), 0j)
        if epsilon < 0 or not np.isfinite(epsilon):
            raise ValueError('Invalid drive')
        upper = (2 * epsilon / kappa) ** 2
        if upper > self.photons[-1]:
            raise ValueError('Drive exceeds nonlinear cavity table domain')
        grid = np.r_[self.photons[self.photons < upper], upper]

        def residual(n):
            return n * ((frequency - self.frequency(n, state)) ** 2 + (kappa / 2) ** 2) - epsilon ** 2
        values = residual(grid)
        crossings = np.flatnonzero((values[:-1] <= 0) & (values[1:] >= 0))
        if not len(crossings):
            n = upper
        else:
            j = crossings[0]
            n = brentq(residual, grid[j], grid[j + 1], xtol=1e-11)
        fr = float(self.frequency(n, state))
        beta = epsilon / (kappa / 2 + 1j * (frequency - fr))
        return (float(n), fr, beta)

@lru_cache(maxsize=32)
def cavity(fq, anharm, fr, g, levels):
    return DressedCavity(fq, anharm, fr, g, levels)

def apparatus(dev, k):
    p = dev.physics.get('readout', {}).get('nonlinear', [None] * dev.n)[k]
    if p is None:
        return None
    q, r = (dev.qubits[k], dev.resonators[k])
    return (cavity(q.freq, q.anharm, r.freq, p['g'], p['levels']), p['drive_scale'])

def pointer(dev, k, frequency, amplitude):
    model = apparatus(dev, k)
    if model is None:
        return None
    c, scale = model
    r = dev.resonators[k]
    return np.array([amplitude - 0.4 * r.kappa / scale * c.steady(frequency, r.kappa, amplitude * scale, m)[2] for m in range(dev.qubits[k].n_levels)])

def waveforms(dev, k, frequency, amplitude, labels, jumps, edges, drive_duration):
    model = apparatus(dev, k)
    if model is None:
        from .temporal_readout import waveforms as linear
        return amplitude * linear(dev.resonators[k], frequency, labels, jumps, edges, drive_duration)
    c, scale = model
    labels, jumps = (np.asarray(labels, int), np.asarray(jumps, float))
    keys = np.column_stack((labels, np.minimum(jumps, edges[-1])))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    labels, jumps = (unique[:, 0].astype(int), unique[:, 1:])
    r = dev.resonators[k]
    beta = np.zeros(len(labels), complex)
    result = np.empty((len(labels), len(edges) - 1), complex)
    active_tables = c.frequencies[:int(np.max(labels)) + 1]
    max_rate = np.pi * r.kappa + 2 * np.pi * np.max(abs(frequency - active_tables))
    max_step = 0.02 / max_rate
    for col, (left, right) in enumerate(zip(edges, edges[1:])):
        cuts = np.sort(np.column_stack((np.full(len(labels), left), np.clip(jumps[:, 0], left, right), np.clip(jumps[:, 1], left, right), np.full(len(labels), np.clip(drive_duration, left, right)), np.full(len(labels), right))), axis=1)
        integrated = np.zeros(len(labels), complex)
        for a, b in zip(cuts.T[:-1], cuts.T[1:]):
            if not np.any(b > a):
                continue
            mid = (a + b) / 2
            level = np.maximum(0, labels - (jumps[:, 0] <= mid) - (jumps[:, 1] <= mid))
            on = (mid < drive_duration).astype(float)
            steps = max(1, int(np.ceil(np.max(b - a) / max_step)))
            dt = (b - a) / steps

            def rhs(z):
                return 2 * np.pi * scale * amplitude * on - (np.pi * r.kappa + 2j * np.pi * (frequency - c.frequency(abs(z) ** 2, level))) * z
            for _ in range(steps):
                k1 = rhs(beta)
                z2 = beta + dt * k1 / 2
                k2 = rhs(z2)
                z3 = beta + dt * k2 / 2
                k3 = rhs(z3)
                z4 = beta + dt * k3
                k4 = rhs(z4)
                mean = (beta + 2 * z2 + 2 * z3 + z4) / 6
                integrated += dt * (amplitude * on - 0.4 * r.kappa / scale * mean)
                beta += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        result[:, col] = integrated / (right - left)
    return result[inverse]
