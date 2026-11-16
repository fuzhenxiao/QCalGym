from copy import deepcopy
from collections import OrderedDict
from contextlib import contextmanager
import cmath
import hashlib
import numpy as np
from .numerics import apply_propagator as mm
import qutip as qt
from scipy.sparse import csr_matrix
from scipy.special import roots_hermite
from .core import Segment
from .flux_coupler import CouplerModel, FluxPulse, FluxSegment, flux_trace, finite

class CouplerSimulator:

    def __init__(self, dev, solver_options=None, *, mode='fast'):
        self.dev = deepcopy(dev)
        self.models = {False: CouplerModel(dev), True: CouplerModel(dev, full=True)}
        options = dict(solver_options or {})
        allowed = {'flux_dt', 'rtol', 'atol', 'max_step', 'noise_tolerance', 'noise_max_order'}
        if set(options) - allowed:
            raise ValueError(f'Unsupported coupler solver options: {set(options) - allowed}')
        self.opts = {'flux_dt': 0.25, 'rtol': 1e-10, 'atol': 1e-12, 'max_step': 0.5, 'noise_tolerance': 2e-06, 'noise_max_order': 17, **options}
        for k in ('flux_dt', 'rtol', 'atol', 'max_step', 'noise_tolerance'):
            finite(self.opts[k], k, 1e-13, 100.0)
        if self.opts['noise_max_order'] not in (5, 9, 17, 33, 65, 129, 257, 513):
            raise ValueError('noise_max_order must be 5,9,17,33,65,129,257,513')
        if mode not in ('fast', 'exact', 'qutip'):
            raise ValueError('Unknown mode')
        self.mode = mode
        self.n = 2
        self.dims = [3, 3]
        self.D = 9
        self.exact_idle = True
        self.exact_flat = True
        self.cal = None
        self._prefix_states = OrderedDict()
        self._prefix_cache_limit = 128
        self.prefix_cache_hits = 0

    def bias(self):
        _, f, _ = flux_trace(self.dev, self.cal, [], 1.0, dt=1.0)
        return f[0]

    def dressed_frequency(self, k):
        return self.models[False].spectrum(self.bias())['f01'][k]

    def marginal(self, pops, k):
        return np.sum(pops, axis=1 - k)

    def ground(self):
        return qt.tensor(qt.basis(3, 0), qt.basis(3, 0)).proj()

    def _schedule(self, seq):
        t = 0.0
        flux = []
        microwave = []
        extra = []
        for item in seq:
            if isinstance(item, FluxSegment):
                flux.extend(((t, p) for p in item.pulses))
                duration = item.duration
            elif isinstance(item, Segment):
                if type(item) is not Segment:
                    raise ValueError('Legacy effective-coupler or custom Segment subclasses are not flux pulses')
                duration = item.duration
                for pulse in item.pulses:
                    if pulse.line not in (0, 1):
                        raise ValueError('Only data-qubit XY ports are available')
                    finite(pulse.start, 'XY start', 0.0, duration)
                    finite(pulse.duration, 'XY duration', 1e-09, 1000000.0)
                    microwave.append((t + pulse.start, pulse))
                    extra.extend([t + pulse.start, t + pulse.start + pulse.duration])
            else:
                duration = float(item)
                finite(duration, 'idle duration', 0.0, 1000000.0)
            t += duration
            extra.append(t)
        return (t, flux, microwave, extra)

    def _initial(self, model, rho0):
        if rho0 is None:
            return np.outer(model.state(), model.state().conj())
        x = np.asarray(rho0.full() if hasattr(rho0, 'full') else rho0, complex)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim == 1:
            x = np.outer(x, x.conj())
        if x.shape == (9, 9):
            full = np.kron(x, np.diag([1.0, 0.0, 0.0]))
            indices = [np.ravel_multi_index(s, (3, 3, 3)) for s in model.basis]
            x = full[np.ix_(indices, indices)]
        elif x.shape == (27, 27) and model.D == 10:
            indices = [np.ravel_multi_index(s, (3, 3, 3)) for s in model.basis]
            x = x[np.ix_(indices, indices)]
        elif x.shape == (10, 10) and model.D == 27:
            indices = [np.ravel_multi_index(s, (3, 3, 3)) for s in self.models[False].basis]
            out = np.zeros((27, 27), complex)
            out[np.ix_(indices, indices)] = x
            x = out
        if x.shape != (model.D, model.D) or not np.allclose(x, x.conj().T, atol=1e-10) or abs(np.trace(x) - 1) > 1e-09 or (np.linalg.eigvalsh(x).min() < -1e-09):
            raise ValueError('Initial state is invalid or lies outside the selected excitation sector')
        return x

    def _trajectory(self, model, times, flux, microwave, rho, noise, dissipative):
        options = {k: self.opts[k] for k in ('rtol', 'atol', 'max_step')}
        if not microwave:
            return model.evolve(times, flux + noise, rho, dissipative=dissipative, **options)
        c = self.dev.crosstalk_matrix()
        c_ops = []
        if dissipative:
            for k, a in enumerate(model.a):
                if np.isfinite(model.T1[k]):
                    c_ops.append(qt.Qobj(csr_matrix(a / np.sqrt(model.T1[k]))))
                if np.isfinite(model.Tphi[k]):
                    c_ops.append(qt.Qobj(csr_matrix(np.diag(model.occ[:, k]) * np.sqrt(2 / model.Tphi[k]))))
        drives = []
        frame = 5.0
        for start, pulse in microwave:
            weights = np.eye(2, dtype=complex)[:, pulse.line]
            if pulse.source_weights is not None:
                weights = np.asarray([complex(*w) for w in pulse.source_weights])
                if weights.shape != (2,):
                    raise ValueError('Invalid XY source mixing')
            amplitudes = mm(c, weights * np.array([q.drive_scale for q in self.dev.qubits]))
            raising = sum((amplitudes[k] * model.a[k].conj().T for k in (0, 1)))

            def coefficients(p=pulse, start=start):
                last_t, last_frame, last_z = (None, None, None)

                def value(t):
                    nonlocal last_t, last_frame, last_z
                    if t != last_t or frame != last_frame:
                        e, de = p.env_scalar(t - start)
                        env = p.amp * (e + 1j * p.drag * de / (2 * np.pi * p.drag_anharm))
                        last_z = env * cmath.exp(-1j * (2 * np.pi * (p.freq - frame) * t + p.phase))
                        last_t, last_frame = (t, frame)
                    return last_z
                return (lambda t: value(t).real, lambda t: value(t).imag)
            real, imag = coefficients()
            terms = [[qt.Qobj(csr_matrix(np.pi * (raising + raising.conj().T))), real], [qt.Qobj(csr_matrix(1j * np.pi * (raising - raising.conj().T))), imag]]
            drives.append((start, pulse, terms))
        digest = hashlib.sha256(repr((self.dev.to_json(), model.full, model.T1, model.Tphi, self.opts, dissipative)).encode())
        digest.update(np.asarray(noise).tobytes())
        digest.update(rho.tobytes())
        active_ids = [tuple((i for i, (start, p, _) in enumerate(drives) if start < end and start + p.duration > begin)) for begin, end in zip(times[:-1], times[1:])]
        index = 0
        while index < len(flux):
            stop = index + 1
            if not active_ids[index]:
                while stop < len(flux) and (not active_ids[stop]):
                    stop += 1
            else:
                while stop < len(flux) and active_ids[stop] == active_ids[index] and np.array_equal(flux[stop], flux[index]):
                    stop += 1
            begin, end = (times[index], times[stop])
            active = [term for i in active_ids[index] for term in drives[i][2]]
            digest.update(np.asarray(times[index:stop + 1]).tobytes())
            digest.update(np.asarray(flux[index:stop]).tobytes())
            digest.update(repr([(drives[i][0], drives[i][1]) for i in active_ids[index]]).encode())
            key = digest.digest()
            if key in self._prefix_states:
                rho = self._prefix_states[key].copy()
                self._prefix_states.move_to_end(key)
                self.prefix_cache_hits += 1
                index = stop
                continue
            if not active:
                rho = model.evolve(times[index:stop + 1], flux[index:stop] + noise, rho, dissipative=dissipative, **options)
                self._remember_prefix(key, rho)
                index = stop
                continue
            f = flux[index]
            frame = next((p.freq for start, p, _ in drives if start < end and start + p.duration > begin))
            carriers = {p.freq for start, p, _ in drives if start < end and start + p.duration > begin}
            if EXACT_DRIVEN_SAMPLES and stop == index + 1 and (end - begin <= 1.0) and (len(carriers) == 1):
                rotation = np.exp(2j * np.pi * (frame - 5.0) * begin * model.occ.sum(axis=1))
                rotated = rotation[:, None] * rho * rotation.conj()[None, :]
                mid = 0.5 * (begin + end)
                h = 2 * np.pi * model.hamiltonian(f + noise, frame=frame)
                for op, coefficient in active:
                    h = h + coefficient(mid) * op.full()
                rho = _exact_sample(model, h, rotated, end - begin, dissipative)
                rotation = np.exp(-2j * np.pi * (frame - 5.0) * end * model.occ.sum(axis=1))
                rho = rotation[:, None] * rho * rotation.conj()[None, :]
                self._remember_prefix(key, rho)
                index = stop
                continue
            rotation = np.exp(2j * np.pi * (frame - 5.0) * begin * model.occ.sum(axis=1))
            rotated = rotation[:, None] * rho * rotation.conj()[None, :]
            h = [qt.Qobj(csr_matrix(2 * np.pi * model.hamiltonian(f + noise, frame=frame)))] + active
            result = qt.mesolve(h, qt.Qobj(rotated), [begin, end], c_ops=c_ops, options={**options, 'rtol': options['rtol'] * 0.01, 'atol': options['atol'] * 0.01, 'method': 'dop853', 'nsteps': max(500000, int(np.ceil((end - begin) * 1000))), 'normalize_output': False, 'store_states': False, 'store_final_state': True})
            rotation = np.exp(-2j * np.pi * (frame - 5.0) * end * model.occ.sum(axis=1))
            rho = rotation[:, None] * result.final_state.full() * rotation.conj()[None, :]
            self._remember_prefix(key, rho)
            index = stop
        return rho

    def _remember_prefix(self, key, rho):
        self._prefix_states[key] = rho.copy()
        self._prefix_states.move_to_end(key)
        if len(self._prefix_states) > self._prefix_cache_limit:
            self._prefix_states.popitem(last=False)

    @contextmanager
    def scan_cache(self):
        previous = self._prefix_cache_limit
        state_bytes = self.models[True].D ** 2 * np.dtype(complex).itemsize
        self._prefix_cache_limit = max(previous, 64 * 1024 ** 2 // state_bytes)
        try:
            yield
        finally:
            self._prefix_cache_limit = previous
            while len(self._prefix_states) > previous:
                self._prefix_states.popitem(last=False)

    def run(self, seq, rho0=None, return_state=False, *, dissipative=True, full=False, t0=0.0):
        if t0 != 0:
            raise ValueError('Coupler acquisitions use a single complete sequence starting at t=0')
        duration, pulses, microwave, extra = self._schedule(seq)
        shape = np.shape(rho0.full() if hasattr(rho0, 'full') else rho0)
        full = full or bool(microwave) or shape in ((9, 9), (9,), (9, 1), (27, 27), (27,), (27, 1))
        model = self.models[full]
        initial = self._initial(model, rho0)
        if duration:
            times, flux, _ = flux_trace(self.dev, self.cal, pulses, duration, self.opts['flux_dt'])
            refined = np.unique(np.r_[times, extra])
            flux = flux[np.minimum(np.searchsorted(times, (refined[:-1] + refined[1:]) / 2, side='right') - 1, len(flux) - 1)]
            starts = np.r_[True, np.any(flux[1:] != flux[:-1], axis=1)]
            starts |= np.isin(refined[:-1], extra)
            times = np.r_[refined[:-1][starts], refined[-1]]
            flux = flux[starts]
            p = self.dev.physics['flux_coupler']
            sigma = p['noise_std']
            direction = np.asarray(p['noise_direction'])
            previous = None
            orders = [0] if not sigma or not np.any(direction) else [3, 5, 9, 17, 33, 65, 129, 257, 513]
            for order in orders:
                if order > self.opts['noise_max_order']:
                    break
                nodes, weights = ([0.0], [1.0]) if order == 0 else roots_hermite(order)
                rho = np.zeros_like(initial)
                for node, weight in zip(nodes, weights):
                    shift = np.sqrt(2) * sigma * node * direction
                    rho += weight / (np.sqrt(np.pi) if order else 1.0) * self._trajectory(model, times, flux, microwave, initial.copy(), shift, dissipative)
                if order == 0 or (previous is not None and np.linalg.norm(rho - previous, ord='nuc') < self.opts['noise_tolerance']):
                    break
                previous = rho
            else:
                raise RuntimeError('Flux noise quadrature did not converge')
            if order > self.opts['noise_max_order']:
                raise RuntimeError('Flux noise quadrature did not converge')
        else:
            rho = initial
        if abs(np.trace(rho) - 1) > 2e-06 or np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() < -2e-06:
            raise RuntimeError('Coupler integration lost trace/positivity; tighten solver settings')
        _, v = model.dressed(self.bias())
        pops = model.populations(mm(mm(v.conj().T, rho), v))
        lab = np.exp(-2j * np.pi * 5.0 * duration * model.occ.sum(axis=1))
        labrho = lab[:, None] * rho * lab.conj()[None, :]
        indices = [np.ravel_multi_index(s, (3, 3, 3)) for s in model.basis]
        fullrho = np.zeros((27, 27), complex)
        fullrho[np.ix_(indices, indices)] = labrho
        return (pops, qt.Qobj(fullrho, dims=[[3, 3, 3], [3, 3, 3]]), duration) if return_state else pops

    def run_many(self, sequences, rho0=None, return_state=False):
        for seq in sequences:
            yield self.run(seq, rho0=rho0, return_state=return_state)

def channel_score(env, sequence, target, phases=None):
    from .performance import average_channel
    sim = env.sim
    model = sim.models[False]
    _, v = model.dressed(sim.bias())
    basis = v[:, model.comp]
    duration = sim._schedule(sequence)[0]
    phases = np.zeros(2) if phases is None else np.asarray(phases, float)
    target = np.asarray(target, complex)
    if target.shape != (4, 4) or not np.isfinite(target).all() or (not np.allclose(mm(target.conj().T, target), np.eye(4), atol=1e-09)):
        raise ValueError('A unitary two-qubit 4x4 target is required')
    if phases.shape != (2,) or not np.isfinite(phases).all():
        raise ValueError('Two finite local phases required')
    finite(duration, 'duration', 0.0, 1000000.0)
    frame = np.array([sum((s[k] * (2 * np.pi * env.cal.drive_freq[k] * duration + phases[k]) for k in (0, 1))) for s in [(0, 0), (0, 1), (1, 0), (1, 1)]])
    lift = [np.ravel_multi_index(s, (3, 3, 3)) for s in model.basis]

    def evolve(rho):
        initial = mm(mm(basis, rho), basis.conj().T)
        _, out, _ = sim.run(sequence, rho0=initial, return_state=True)
        projected = mm(mm(basis.conj().T, out.full()[np.ix_(lift, lift)]), basis)
        return np.exp(1j * frame)[:, None] * projected * np.exp(-1j * frame)[None, :]
    return {**average_channel(evolve, np.asarray(target, complex)), 'duration_ns': float(duration)}
EXACT_DRIVEN_SAMPLES = True

def _exact_sample(model, h, rho, dt, dissipative):
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply
    D = model.D
    identity = sparse.eye(D, format='csr')
    hs = sparse.csr_matrix(h)
    generator = -1j * (sparse.kron(hs, identity) - sparse.kron(identity, hs.T))
    if dissipative:
        model.dissipator(np.zeros((D, D), complex))
        generator = generator + model._dissipator_csr
    generator = sparse.csr_matrix(generator) * dt
    return expm_multiply(generator, rho.ravel(), traceA=generator.diagonal().sum()).reshape(rho.shape)
