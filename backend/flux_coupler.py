from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
import math
import numpy as np
from .numerics import apply_propagator as mm, expm
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from .device import DeviceConfig, CalibrationStore, QubitParams, CouplingParams, ResonatorParams
from .hardware import flux_frequency
KINDS = ('flux_coupler_tunable_transmon', 'flux_coupler_fixed_transmon')

@dataclass
class FluxCouplerCalibration(CalibrationStore):
    coupler_flux_bias: dict = field(default_factory=dict)
    flux_mix: dict = field(default_factory=dict)
    flux_inverse_tau: dict = field(default_factory=dict)
    flux_inverse_tail: dict = field(default_factory=dict)
    native_flux_gate: dict = field(default_factory=dict)

def flux_law(fmax, ec=0.25, asymmetry=0.45):
    return dict(ej_max=(fmax + ec) ** 2 / (8 * ec), ec=ec, asymmetry=asymmetry, offset=0.0, bias_min=-0.45, bias_max=0.45)

def make_flux_coupler_device(kind=KINDS[0], seed=0):
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    fs = [5.4, 5.2] if kind == KINDS[0] else [5.0, 5.2]
    laws = [flux_law(f) for f in fs] if kind == KINDS[0] else [None, None]
    p = dict(kind=kind, qubit_flux=laws, coupler_flux=flux_law(7.0, 0.2, 0.3), g=[0.08, 0.085], coupling_reference=fs + [7.0], coupler_T1=30000.0, coupler_Tphi=60000.0, flux_matrix=np.eye(3).tolist(), line_tau=[0.0, 0.0, 0.0], noise_std=0.0, noise_direction=[0.0, 0.0, 1.0])
    return DeviceConfig([QubitParams(f, -0.25, 60000.0, 100000.0) for f in fs], [CouplingParams(0, 1, 0.0045)], [ResonatorParams(7.6 + 0.1 * k, 0.001, -0.0005) for k in range(2)], seed=seed, name=kind, topology='pair', physics={'flux_coupler': p})

def finite(x, name, low=-np.inf, high=np.inf):
    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x) or (not np.isfinite(x)) or (not low <= x <= high):
        raise ValueError(f'{name} must be finite in [{low}, {high}]')

def validate_device(dev):
    if 'flux_coupler' not in dev.physics or set(dev.physics) - {'flux_coupler', 'readout'} or dev.n != 2 or any((q.n_levels != 3 for q in dev.qubits)):
        raise ValueError('Flux coupler requires exactly two qutrit data qubits; other optional modules cannot yet be combined')
    from .nonlinear_readout import validate_readout
    validate_readout(dev)
    p = dev.physics['flux_coupler']
    allowed = {'kind', 'qubit_flux', 'coupler_flux', 'g', 'coupling_reference', 'coupler_T1', 'coupler_Tphi', 'flux_matrix', 'line_tau', 'noise_std', 'noise_direction'}
    if set(p) - {'line_tail'} != allowed or p['kind'] not in KINDS:
        raise ValueError('Unknown/missing flux-coupler hardware field')
    tail = np.asarray(p.get('line_tail', [1.0, 1.0, 1.0]), float)
    if tail.shape != (3,) or not np.isfinite(tail).all() or tail.min() < 0 or (tail.max() > 1):
        raise ValueError('line_tail requires three fractions in [0, 1]')
    if len(dev.resonators) != 2 or len(dev.couplings) != 1 or (dev.couplings[0].i, dev.couplings[0].j) != (0, 1):
        raise ValueError('Exactly one direct 0-1 edge and two readout resonators are required')
    finite(dev.couplings[0].g, 'direct coupling', -0.2, 0.2)
    if len(p['qubit_flux']) != 2:
        raise ValueError('qubit_flux must have two entries')
    tunable = p['kind'] == KINDS[0]
    if any(((f is not None) != tunable for f in p['qubit_flux'])):
        raise ValueError('Hardware kind and qubit flux capabilities disagree')
    for f in p['qubit_flux'] + [p['coupler_flux']]:
        if f is None:
            continue
        if set(f) != {'ej_max', 'ec', 'asymmetry', 'offset', 'bias_min', 'bias_max'}:
            raise ValueError('Unknown/missing SQUID law field')
        for key in ('ej_max', 'ec'):
            finite(f[key], key, 1e-05, 200.0)
        finite(f['asymmetry'], 'asymmetry', 0.01, 1.0)
        finite(f['offset'], 'offset', -0.5, 0.5)
        finite(f['bias_min'], 'bias_min', -0.5, 0.5)
        finite(f['bias_max'], 'bias_max', -0.5, 0.5)
        if f['bias_min'] >= f['bias_max']:
            raise ValueError('Empty flux interval')
    for key, size in [('g', 2), ('coupling_reference', 3), ('line_tau', 3), ('noise_direction', 3)]:
        a = np.asarray(p[key], float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f'Invalid {key}')
    if min(p['coupling_reference']) <= 0 or min(p['line_tau']) < 0 or max(p['line_tau']) > 100000.0:
        raise ValueError('Invalid coupling reference / line time constant')
    finite(p['noise_std'], 'noise_std', 0.0, 0.01)
    m = np.asarray(p['flux_matrix'], float)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError('Invalid flux matrix')
    for t in [q.T1 for q in dev.qubits] + [q.Tphi for q in dev.qubits] + [p['coupler_T1'], p['coupler_Tphi']]:
        if np.isnan(t) or t <= 0:
            raise ValueError('Coherence time must be positive (infinity allowed)')
    for q in dev.qubits:
        finite(q.freq, 'qubit frequency', 0.1, 15.0)
        finite(q.anharm, 'anharmonicity', -2.0, -0.001)

@dataclass(frozen=True)
class FluxPulse:
    port: int
    amp: float
    duration: float
    rise: float = 0.0
    freq: float = 0.0
    phase: float = 0.0

    def __post_init__(self):
        if type(self.port) is not int or self.port not in (0, 1, 2):
            raise ValueError('Invalid flux port')
        finite(self.amp, 'flux amplitude', -1.0, 1.0)
        finite(self.duration, 'duration', 1e-09, 1000000.0)
        finite(self.rise, 'rise', 0.0, self.duration / 2)
        finite(self.freq, 'modulation frequency', 0.0, 5.0)
        finite(self.phase, 'phase')

    def value(self, t):
        if t < 0 or t >= self.duration:
            return 0.0
        edge = min(t, self.duration - t)
        env = 0.5 * (1 - np.cos(np.pi * edge / self.rise)) if self.rise and edge < self.rise else 1.0
        return self.amp * env * np.cos(2 * np.pi * self.freq * t + self.phase)

@dataclass(frozen=True)
class FluxSegment:
    pulses: tuple
    duration: float

    def __post_init__(self):
        finite(self.duration, 'segment duration', 1e-09, 1000000.0)
        if any((not isinstance(p, FluxPulse) or p.duration > self.duration for p in self.pulses)):
            raise ValueError('Flux pulses must fit inside their segment')

class CouplerModel:

    def __init__(self, dev, *, full=False):
        validate_device(dev)
        self.dev = deepcopy(dev)
        self.p = self.dev.physics['flux_coupler']
        self.full = full
        self.basis = [s for s in product(range(3), repeat=3) if full or sum(s) <= 2]
        self.index = {s: k for k, s in enumerate(self.basis)}
        self.D = len(self.basis)
        self.occ = np.array(self.basis, float)
        self.a = []
        for k in range(3):
            a = np.zeros((self.D, self.D), complex)
            for j, s in enumerate(self.basis):
                if s[k]:
                    dest = list(s)
                    dest[k] -= 1
                    a[self.index[tuple(dest)], j] = np.sqrt(s[k])
            self.a.append(a)
        self.exchanges = [mm(self.a[i].conj().T, self.a[j]) + mm(self.a[j].conj().T, self.a[i]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        self.alpha = np.array([-f['ec'] if f else q.anharm for q, f in zip(dev.qubits, self.p['qubit_flux'])] + [-self.p['coupler_flux']['ec']])
        self.T1 = [q.T1 for q in dev.qubits] + [self.p['coupler_T1']]
        self.Tphi = [q.Tphi for q in dev.qubits] + [self.p['coupler_Tphi']]
        self.comp = [self.index[s] for s in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]]
        self._lowering = []
        for a in self.a:
            dest, source = np.nonzero(a)
            self._lowering.append((np.ix_(dest, dest), np.ix_(source, source), a[dest, source, None] * a[dest, source].conj()[None, :]))
        self._dissipation_key = None

    def frequencies(self, flux):
        if np.shape(flux) != (3,) or not np.isfinite(flux).all():
            raise ValueError('Expected three finite loop fluxes')
        out = []
        for k, law in enumerate(self.p['qubit_flux'] + [self.p['coupler_flux']]):
            if law is None:
                out.append(self.dev.qubits[k].freq)
            else:
                finite(flux[k], 'physical flux', law['bias_min'], law['bias_max'])
                x = np.pi * (flux[k] + law['offset'])
                ej = law['ej_max'] * np.sqrt(np.cos(x) ** 2 + law['asymmetry'] ** 2 * np.sin(x) ** 2)
                if ej / law['ec'] < 20:
                    raise ValueError('EJ/EC<20: outside supported Duffing-transmon range')
                out.append(flux_frequency(law, flux[k]))
        return np.asarray(out)

    def hamiltonian(self, flux, frame=5.0, *, counter_rotating=False):
        f = self.frequencies(flux)
        diagonal = mm(self.occ, f - frame) + mm(0.5 * self.occ * (self.occ - 1), self.alpha)
        h = np.diag(diagonal).astype(complex)
        refs = np.asarray(self.p['coupling_reference'])
        for (i, j), g, ex in zip([(0, 1), (0, 2), (1, 2)], [self.dev.couplings[0].g] + self.p['g'], self.exchanges):
            scale = g * np.sqrt(f[i] * f[j] / (refs[i] * refs[j]))
            h += scale * ex
            if counter_rotating:
                if not self.full or frame != 0:
                    raise ValueError('Counter-rotating audit requires full lab-frame model')
                h -= scale * (mm(self.a[i], self.a[j]) + mm(self.a[i].conj().T, self.a[j].conj().T))
        return h

    def dressed(self, flux):
        e, v = eigh(self.hamiltonian(flux, frame=0))
        rows, cols = linear_sum_assignment(-np.abs(v) ** 2)
        v, e = (v[:, cols], e[cols])
        phase = np.exp(-1j * np.angle(np.diag(v)))
        return (e, v * phase)

    def spectrum(self, flux):
        e, v = self.dressed(flux)
        c = self.comp
        one = [self.index[1, 0, 0], self.index[0, 1, 0]]
        b = v[np.ix_(one, one)]
        u, _, vh = np.linalg.svd(b)
        rot = mm(u, vh)
        heff = mm(mm(rot, np.diag(e[one])), rot.conj().T)
        return dict(f01=[float(e[c[2]] - e[c[0]]), float(e[c[1]] - e[c[0]])], zz=float(e[c[3]] - e[c[2]] - e[c[1]] + e[c[0]]), exchange=float(heff[0, 1].real), min_label_overlap=float(min(np.abs(np.diag(v))[c] ** 2)))

    def state(self, label=(0, 0, 0), *, dressed_flux=None):
        v = np.eye(self.D, dtype=complex)[:, self.index[tuple(label)]]
        return v if dressed_flux is None else mm(self.dressed(dressed_flux)[1], v)

    def populations(self, rho):
        out = np.zeros((3, 3))
        for p, s in zip(np.diag(rho).real, self.basis):
            out[s[0], s[1]] += p
        return out

    def dissipator(self, rho):
        key = (tuple(self.T1), tuple(self.Tphi))
        if key != self._dissipation_key:
            decay = np.zeros((self.D, self.D))
            gains = []
            for k, (dest, source, factor) in enumerate(self._lowering):
                n = self.occ[:, k]
                if np.isfinite(self.T1[k]):
                    decay += 0.5 * (n[:, None] + n[None, :]) / self.T1[k]
                    gains.append((dest, source, factor / self.T1[k]))
                if np.isfinite(self.Tphi[k]):
                    decay += (n[:, None] - n[None, :]) ** 2 / self.Tphi[k]
            self._decay, self._gains, self._dissipation_key = (decay, gains, key)
            from scipy.sparse import coo_matrix
            ids = np.arange(self.D * self.D)
            rows = [ids]
            cols = [ids]
            data = [-decay.ravel()]
            for dest, source, factor in gains:
                rows.append((dest[0] * self.D + dest[1]).ravel())
                cols.append((source[0] * self.D + source[1]).ravel())
                data.append(factor.ravel())
            self._dissipator_csr = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(self.D * self.D, self.D * self.D)).tocsr()
        return (self._dissipator_csr @ rho.ravel()).reshape(rho.shape)

    def _static_open_step(self, h, rho, duration):
        from scipy import sparse
        key = (h.tobytes(), tuple(self.T1), tuple(self.Tphi))
        if getattr(self, '_static_generator_key', None) != key:
            self.dissipator(np.zeros_like(rho))
            identity = sparse.eye(self.D, format='csr')
            hs = sparse.csr_matrix(h)
            generator = -1j * (sparse.kron(hs, identity) - sparse.kron(identity, hs.T))
            generator += sparse.diags(-self._decay.ravel())
            for k, a in enumerate(self.a):
                if np.isfinite(self.T1[k]):
                    lower = sparse.csr_matrix(a)
                    generator += sparse.kron(lower, lower.conj()) / self.T1[k]
            generator = generator.tocsr()
            excitation = self.occ.sum(axis=1)
            differences = (excitation[:, None] - excitation[None, :]).ravel()
            self._static_blocks = []
            for difference in np.unique(differences):
                ids = np.flatnonzero(differences == difference)
                self._static_blocks.append((ids, generator[ids][:, ids].toarray()))
            self._static_generator_key = key
            self._static_duration = None
        if self._static_duration != duration:
            self._static_propagators = {}
            self._static_duration = duration
        result = np.zeros(rho.size, complex)
        values = rho.ravel()
        for block_index, (ids, generator) in enumerate(self._static_blocks):
            if not np.any(values[ids]):
                continue
            if block_index not in self._static_propagators:
                self._static_propagators[block_index] = expm(generator * duration)
            result[ids] = mm(self._static_propagators[block_index], values[ids])
        return result.reshape(rho.shape)

    def _sample_generator(self, h, duration):
        from scipy import sparse
        self.dissipator(np.zeros((self.D, self.D), complex))
        key = self._dissipation_key
        if getattr(self, '_sample_structure_key', None) != key:
            identity = sparse.eye(self.D, format='csr')
            pattern = sparse.csr_matrix(np.eye(self.D) + sum((abs(v) for v in self.exchanges)))
            structure = (sparse.kron(pattern, identity, format='csr') + sparse.kron(identity, pattern.T, format='csr') + abs(self._dissipator_csr)).tocsr()
            rows = np.repeat(np.arange(self.D ** 2), np.diff(structure.indptr))
            cols = structure.indices
            i, j = (rows // self.D, rows % self.D)
            k, l = (cols // self.D, cols % self.D)
            dissipation = np.asarray(self._dissipator_csr[rows, cols]).ravel()
            self._sample_structure = (structure.indptr, cols, i, j, k, l, dissipation)
            self._sample_structure_key = key
        indptr, indices, i, j, k, l, dissipation = self._sample_structure
        data = duration * (-1j * (h[i, k] * (j == l) - h[l, j] * (i == k)) + dissipation)
        return sparse.csr_matrix((data, indices, indptr), shape=(self.D ** 2, self.D ** 2))

    def evolve(self, times, flux, initial, *, dissipative=True, rtol=1e-10, atol=1e-12, max_step=0.5):
        times, flux = (np.asarray(times, float), np.asarray(flux, float))
        if times.ndim != 1 or len(times) < 2 or (not np.isfinite(times).all()) or np.any(np.diff(times) <= 0):
            raise ValueError('Strictly increasing finite times required')
        if flux.shape != (len(times) - 1, 3):
            raise ValueError('One flux vector per DAC interval required')
        x = np.asarray(initial, complex).copy()
        if x.shape == (self.D,):
            x = np.outer(x, x.conj())
        if x.shape != (self.D, self.D):
            raise ValueError('Initial state has wrong dimension')
        if not np.allclose(x, x.conj().T, atol=1e-10) or abs(np.trace(x) - 1) > 1e-09 or np.linalg.eigvalsh(x).min() < -1e-09:
            raise ValueError('Initial state must be a normalized positive density matrix')
        starts = np.r_[True, np.any(flux[1:] != flux[:-1], axis=1)]
        times = np.r_[times[:-1][starts], times[-1]]
        flux = flux[starts]
        for dt, f in zip(np.diff(times), flux):
            h = 2 * np.pi * self.hamiltonian(f)
            if not dissipative or all((np.isinf(t) for t in self.T1 + self.Tphi)):
                e, v = eigh(h)
                u = mm(v * np.exp(-1j * e * dt), v.conj().T)
                x = mm(mm(u, x), u.conj().T)
            elif dt >= 10.0:
                x = self._static_open_step(h, x, float(dt))
            else:
                from scipy.sparse.linalg import expm_multiply
                generator = self._sample_generator(h, dt)
                x = expm_multiply(generator, x.ravel(), traceA=generator.diagonal().sum()).reshape(x.shape)
        return x

def flux_trace(dev, cal, pulses, duration, dt=0.25):
    validate_device(dev)
    finite(duration, 'duration', 1e-09, 1000000.0)
    finite(dt, 'dt', 0.001, 100.0)
    p = dev.physics['flux_coupler']
    fixed = p['kind'] == KINDS[1]
    bias = np.array([cal.flux_bias.get(k, 0.0) for k in (0, 1)] + [cal.coupler_flux_bias.get(0, 0.0)])
    inverse_tail_map = getattr(cal, 'flux_inverse_tail', {})
    if set(cal.flux_bias) - {0, 1} or set(cal.coupler_flux_bias) - {0} or set(cal.flux_mix) - {0} or set(cal.flux_inverse_tau) - {0, 1, 2} or set(inverse_tail_map) - {0, 1, 2}:
        raise ValueError('Invalid flux control index')
    mix = np.asarray(cal.flux_mix.get(0, np.eye(3)), float)
    inverse = np.array([cal.flux_inverse_tau.get(k, 0.0) for k in range(3)], float)
    inverse_tail = np.array([inverse_tail_map.get(k, 1.0) for k in range(3)], float)
    if mix.shape != (3, 3) or not np.isfinite(mix).all() or (not np.isfinite(bias).all()) or (not np.isfinite(inverse).all()) or (min(inverse) < 0) or (not np.isfinite(inverse_tail).all()) or (inverse_tail.min() <= 0) or (inverse_tail.max() > 1):
        raise ValueError('Invalid flux controls')
    if fixed and (np.any(bias[:2]) or any((pulse.port != 2 for _, pulse in pulses)) or np.any(mix[:2, 2]) or np.any(inverse[:2])):
        raise ValueError('Fixed-frequency hardware has only coupler Z port 2')
    boundaries = [0.0, duration]
    for start, pulse in pulses:
        if not isinstance(pulse, FluxPulse):
            raise ValueError('Expected FluxPulse')
        if pulse.freq * dt > 0.1:
            raise ValueError('Flux modulation requires at least 10 DAC samples per period; reduce flux_dt and check convergence')
        finite(start, 'pulse start', 0.0, duration)
        if start + pulse.duration > duration + 1e-09:
            raise ValueError('Pulse exceeds sequence duration')
        boundaries.extend([start, start + pulse.duration])
    times = np.unique(np.r_[np.arange(0, duration, dt), boundaries]) if pulses else np.array([0.0, duration])
    mid = (times[:-1] + times[1:]) / 2
    command = np.tile(bias, (len(mid), 1))
    for start, pulse in pulses:
        command[:, pulse.port] += [pulse.value(t - start) for t in mid]
    physical = np.empty_like(command)
    previous = bias.copy()
    matrix = np.asarray(p['flux_matrix'])
    memory = mm(matrix, mm(mix, bias))
    tau = np.asarray(p['line_tau'])
    safe = np.where(tau > 0, tau, 1.0)
    tail = np.asarray(p.get('line_tail', [1.0, 1.0, 1.0]), float)
    inverse_state = bias.copy()
    model = CouplerModel(dev)
    model.frequencies(memory)
    has_filter = bool(np.any(tau > 0))
    for i, step in enumerate(np.diff(times)):
        ah = np.exp(-step / np.where(inverse > 0, inverse, 1.0))
        A = inverse_tail
        digital = np.where(inverse > 0, (command[i] - A * ah * inverse_state) / (1 - A + A * (1 - ah)), command[i])
        inverse_state = np.where(inverse > 0, (1 - ah) * digital + ah * inverse_state, command[i])
        previous = command[i].copy()
        target = mm(matrix, mm(mix, digital))
        lowpass = target + (memory - target) * np.exp(-step / (2 * safe))
        physical[i] = np.where(tau > 0, (1 - tail) * target + tail * lowpass, target)
        memory = np.where(tau > 0, target + (memory - target) * np.exp(-step / safe), target)
        if has_filter:
            model.frequencies(memory)
    for f in physical:
        model.frequencies(f)
    return (times, physical, command)
