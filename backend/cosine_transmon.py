from functools import lru_cache
from collections import OrderedDict
from hashlib import sha256
import numpy as np
import qutip as qt
from scipy.linalg import eigh_tridiagonal

@lru_cache(maxsize=32)
def spectrum(ej, ec, ng, cutoff, levels):
    charge = np.arange(-cutoff, cutoff + 1)
    energy, vectors = eigh_tridiagonal(4 * ec * (charge - ng) ** 2, np.full(2 * cutoff, -ej / 2))
    vectors = vectors[:, :levels]
    matrix = np.einsum('ij,i,ik->jk', vectors, charge, vectors, optimize=False)
    signs = np.ones(levels)
    for j in range(1, levels):
        signs[j] = signs[j - 1] * np.sign(matrix[j - 1, j])
    matrix = signs[:, None] * matrix * signs[None, :]
    energy = energy[:levels]
    return (energy, matrix / abs(matrix[0, 1]))

def validate(dev):
    from .hardware import _number
    p = dev.physics.get('cosine_transmon')
    if p is None:
        return
    if dev.n != 1 or dev.couplings:
        raise ValueError('Full cosine backend currently supports one directly driven transmon')
    if not isinstance(p, dict) or set(p) != {'ej', 'ec', 'ng', 'charge_cutoff'}:
        raise ValueError('Cosine transmon requires ej, ec, ng, charge_cutoff')
    for key in ('ej', 'ec'):
        _number(p[key], key, 1e-06)
    _number(p['ng'], 'offset charge', -0.5, 0.5)
    cutoff = p['charge_cutoff']
    levels = dev.qubits[0].n_levels
    if type(cutoff) != int or not 12 <= cutoff <= 100 or (not 8 <= levels <= min(32, 2 * cutoff - 3)):
        raise ValueError('Cosine model needs 8–32 eigenstates and a larger charge basis')
    if set(dev.physics) - {'cosine_transmon', 'readout', 'mixers', 'delays', 'delay_kind', 'ampm', 'amplitude_compression'}:
        raise ValueError('Full cosine backend supports memoryless source controls, not auxiliary bath models')
    energies, _ = spectrum(p['ej'], p['ec'], p['ng'], cutoff, levels)
    if energies[-2] <= p['ej']:
        raise ValueError('Truncation does not retain states above the cosine barrier')
    if abs(dev.qubits[0].freq - (energies[1] - energies[0])) > 1e-08:
        raise ValueError('Qubit f01 must match the cosine Hamiltonian')

def operators(sim):
    p = sim.dev.physics['cosine_transmon']
    energy, charge = spectrum(p['ej'], p['ec'], p['ng'], p['charge_cutoff'], sim.D)
    sim.cosine_energies = energy.copy()
    sim.charge = qt.Qobj(charge, dims=[sim.dims] * 2)
    sim.a = [qt.Qobj(np.triu(charge, 1), dims=[sim.dims] * 2)]
    sim.ad = [sim.a[0].dag()]
    sim.num = [qt.Qobj(np.diag(np.arange(sim.D)), dims=[sim.dims] * 2)]
    sim.X = [sim.a[0] + sim.ad[0]]
    sim.Y = [1j * (sim.a[0] - sim.ad[0])]

def static(sim):
    sim.H_lin = 2 * np.pi * sim.dev.qubits[0].freq * sim.num[0]
    sim.H_static = qt.Qobj(np.diag(2 * np.pi * (sim.cosine_energies - sim.cosine_energies[0])), dims=[sim.dims] * 2)
    sim.H_anh = sim.H_static - sim.H_lin
    sim.H_c = 0 * sim.H_static
    sim.coup_terms = []
    q = sim.dev.qubits[0]
    sim.c_ops = []
    dipole = sim.a[0].full()
    if np.isfinite(q.T1):
        for i, j in zip(*np.nonzero(abs(dipole) > 1e-10)):
            op = np.zeros((sim.D, sim.D), complex)
            op[i, j] = dipole[i, j] / np.sqrt(q.T1)
            sim.c_ops.append(qt.Qobj(op, dims=[sim.dims] * 2))
    if np.isfinite(q.Tphi):
        sim.c_ops.append(np.sqrt(2 / q.Tphi) * sim.num[0])
    sim.w_frame = np.array([2 * np.pi * q.freq])

def apparatus_active(sim):
    from .hardware import APPARATUS_CONTROLS
    controls = getattr(sim, 'hardware_controls', {})
    return bool({'mixers', 'delays', 'ampm', 'amplitude_compression'} & sim.dev.physics.keys()) or any((controls.get(name) for names in APPARATUS_CONTROLS.values() for name in names))

class SourceWaveform:

    def __init__(self, sim, commands, head, t0):
        from .extended_dynamics import source_signals, iq_matrix, sidebands
        self.signals = source_signals(sim, commands, head, t0, 0.0)
        self.gain = 2 * np.pi * sim.dev.qubits[0].drive_scale * sim.dev.crosstalk_matrix()[0, 0]
        controls = getattr(sim, 'hardware_controls', {})
        amplitude = 0.0
        for pulse, weight, _, _ in commands:
            feature = pulse.duration / pulse.n_sigma if pulse.shape == 'gaussian' else min(pulse.rise, pulse.duration / 2) if pulse.shape == 'gaussian_square' else pulse.duration
            derivative = 4 / max(feature, 1e-12) if pulse.shape != 'square' else 0.0
            drag_bound = abs(pulse.drag) * derivative / (2 * np.pi * abs(pulse.drag_anharm)) if pulse.drag else 0.0
            amplitude += abs(pulse.amp * weight) * (1 + drag_bound)
        mixer = sim.dev.physics.get('mixers')
        lo = controls.get('lo_freq', {}).get(0, 5.0)
        frequencies = [p.freq for p, _, _, _ in commands]
        if mixer:
            physical = iq_matrix(mixer[0].get('gain', 1.0), mixer[0].get('skew', 0.0))
            main, image = sidebands(physical @ iq_matrix(controls.get('mixer_gain', {}).get(0, 1.0), controls.get('mixer_skew', {}).get(0, 0.0)))
            dc = physical @ np.array([controls.get('mixer_dc_i', {}).get(0, 0.0), controls.get('mixer_dc_q', {}).get(0, 0.0)])
            leak = complex(*mixer[0].get('leakage', [0.0, 0.0])) + complex(*dc)
            amplitude = (abs(main) + abs(image)) * amplitude + abs(leak)
            frequencies += [lo] + [abs(2 * lo - f) for f in frequencies] if image else [lo]
        self.bound = abs(self.gain) * amplitude
        self.max_frequency = max(frequencies or [0.0])

    def __call__(self, times):

        def scalar(t):
            return sum(((self.gain * complex(real(float(t)), imag(float(t)))).real for _, real, imag in self.signals))
        values = np.asarray(times)
        if values.ndim == 0:
            return scalar(float(values))
        return np.fromiter((scalar(t) for t in values.flat), dtype=float, count=values.size).reshape(values.shape)

def controlled_schedule(sim, sequence):
    from .core import Pulse
    from .extended_dynamics import schedule
    commands, total, head = schedule(sim, sequence)
    controls = getattr(sim, 'hardware_controls', {})
    amplitude = controls.get('dressing_amp', {}).get(0, 0.0)
    if amplitude and total > 0:
        pulse = Pulse(0, controls.get('dressing_freq', {}).get(0, 5.2), amplitude, total, shape='square')
        delay = sim.dev.physics.get('delays', [0.0])[0] + controls.get('channel_delay', {}).get(0, 0.0)
        commands.append((pulse, 1.0 + 0j, 0.0, delay))
    return (commands, total, head)

def drive_hamiltonian(sim, pulses, start=0.0):
    if callable(pulses):
        return [sim.H_static, [sim.charge, lambda t: float(pulses(t))]]
    if apparatus_active(sim):
        from .core import Segment
        commands, _, head = controlled_schedule(sim, [Segment(pulses)])
        waveform = SourceWaveform(sim, commands, head, start)
        return [sim.H_static, [sim.charge, lambda t: float(waveform(t))]]

    def coefficient(t):
        value = 0.0
        for pulse in pulses:
            e, de = pulse.env_scalar(t - start - pulse.start)
            z = complex(e, pulse.drag * de / (2 * np.pi * pulse.drag_anharm) if pulse.drag else 0.0)
            weight = complex(*pulse.source_weights[0]) if pulse.source_weights else 1.0
            value += (weight * pulse.amp * z * np.exp(1j * (2 * np.pi * pulse.freq * t + pulse.phase))).real
        return 2 * np.pi * sim.dev.qubits[0].drive_scale * value
    return [sim.H_static, [sim.charge, coefficient]]
FAST_PROPAGATION = True
FAST_DT_NS = 0.004
FAST_PHASE_STEP = 0.08
_Y1 = 1 / (2 - 2 ** (1 / 3))
_Y0 = 1 - 2 * _Y1
_cache = {}

def _static_superop(sim):
    key = (sim, 'L0')
    if key not in _cache:
        if len(_cache) >= 256:
            _cache.clear()
        D = sim.D
        I = np.eye(D)
        h = sim.H_static.full()
        Ld = np.zeros((D * D, D * D), complex)
        for c in sim.c_ops:
            c = c.full()
            cd = c.conj().T
            cdc = cd @ c
            Ld += np.kron(c, c.conj()) - 0.5 * (np.kron(cdc, I) + np.kron(I, cdc.T))
        L = -1j * (np.kron(h, I) - np.kron(I, h.T)) + Ld
        lam, V = np.linalg.eigh(sim.charge.full())
        _cache[key] = (L, lam, V, Ld)
    return _cache[key]

def _superop_exp(sim, dt):
    key = (sim, 'P', round(dt, 15))
    if key not in _cache:
        from scipy.linalg import expm
        if len(_cache) >= 256:
            _cache.clear()
        _cache[key] = expm(_static_superop(sim)[0] * dt)
    return _cache[key]

def _coefficients(sim, pulses, start, times):
    if callable(pulses):
        return pulses(times)
    value = np.zeros(len(times))
    for pulse in pulses:
        env = np.array([pulse.env_scalar(t - start - pulse.start) for t in times])
        e, de = (env[:, 0], env[:, 1])
        z = e + 1j * (pulse.drag * de / (2 * np.pi * pulse.drag_anharm) if pulse.drag else 0.0)
        weight = complex(*pulse.source_weights[0]) if pulse.source_weights else 1.0
        value += (weight * pulse.amp * z * np.exp(1j * (2 * np.pi * pulse.freq * times + pulse.phase))).real
    return 2 * np.pi * sim.dev.qubits[0].drive_scale * value

def _fast_driven_stepwise(sim, pulses, clock, duration, rho, dt_target=None):
    D = sim.D
    _, lam, V, _ = _static_superop(sim)
    steps, dt = _step_size(sim, pulses, duration, dt_target)
    weights = np.array([_Y1, _Y0, _Y1])
    offsets = np.array([_Y1 / 2, _Y1 + _Y0 / 2, _Y1 + _Y0 + _Y1 / 2])
    mids = (clock + dt * (np.arange(steps)[:, None] + offsets[None, :])).ravel()
    c = _coefficients(sim, pulses, clock, mids).reshape(steps, 3)
    phases = np.exp(-1j * (c * weights[None, :] * dt)[..., None] * lam[None, None, :])
    U = np.einsum('ij,skj,lj->skil', V, phases, V.conj(), optimize=True)
    half = [_superop_exp(sim, w * dt / 2) for w in (_Y1, _Y0)]
    edge = [half[0], half[0] @ half[1], half[1] @ half[0], half[0] @ half[0]]
    batch = rho.ndim == 3
    K = rho.shape[0] if batch else 1
    r = rho.reshape(K, D * D).T
    r = edge[0] @ r
    for k in range(steps):
        for j in range(3):
            m = r.T.reshape(K, D, D)
            m = U[k, j] @ m @ U[k, j].conj().T
            r = m.reshape(K, D * D).T
            if j == 0:
                r = edge[1] @ r
            elif j == 1:
                r = edge[2] @ r
        r = (edge[3] if k < steps - 1 else half[0]) @ r
    r = r.T.reshape(K, D, D)
    return r if batch else r[0]
FAST_DISS_CHUNK = 0.02
FAST_BLOCK_STEPS = 2048

def _step_size(sim, pulses, duration, dt_target=None):
    lam = _static_superop(sim)[1]
    if callable(pulses):
        bound = pulses.bound
        dt_max = min(dt_target or FAST_DT_NS, 0.125 / (2 * np.pi * max(pulses.max_frequency, 1e-12)))
    else:
        bound = 2 * np.pi * sim.dev.qubits[0].drive_scale * sum((abs(p.amp) * (abs(complex(*p.source_weights[0])) if p.source_weights else 1.0) * (1 + abs(p.drag)) for p in pulses))
        dt_max = dt_target or FAST_DT_NS
    if bound > 0:
        dt_max = min(dt_max, FAST_PHASE_STEP / (bound * np.max(abs(lam))))
    steps = max(1, int(np.ceil(duration / dt_max)))
    return (steps, duration / steps)

def _dissipator_exp(sim, t):
    key = (sim, 'Ld', round(t, 15))
    if key not in _cache:
        from scipy.linalg import expm
        if len(_cache) >= 256:
            _cache.clear()
        _cache[key] = expm(_static_superop(sim)[3] * t)
    return _cache[key]

def _fast_driven(sim, pulses, clock, duration, rho, dt_target=None):
    D = sim.D
    _, lam, V, _ = _static_superop(sim)
    steps, dt = _step_size(sim, pulses, duration, dt_target)
    m = max(1, min(steps, int(round(FAST_DISS_CHUNK / dt))))
    energy = np.diag(sim.H_static.full()).real
    weights = np.array([_Y1, _Y0, _Y1])
    offsets = np.array([_Y1 / 2, _Y1 + _Y0 / 2, _Y1 + _Y0 + _Y1 / 2])
    half_phase = np.exp(-1j * energy[None, :] * weights[:, None] * dt / 2)
    batch = rho.ndim == 3
    K = rho.shape[0] if batch else 1
    r = rho.reshape(K, D * D).T
    eye = np.eye(D, dtype=complex)
    block = max(m, FAST_BLOCK_STEPS // m * m)
    for s0 in range(0, steps, block):
        s1 = min(steps, s0 + block)
        n = s1 - s0
        mids = (clock + dt * (np.arange(s0, s1)[:, None] + offsets[None, :])).ravel()
        c = _coefficients(sim, pulses, clock, mids).reshape(n, 3)
        phases = np.exp(-1j * (c * weights[None, :] * dt)[..., None] * lam[None, None, :])
        U = np.einsum('ij,skj,lj->skil', V, phases, V.conj(), optimize=True)
        U = half_phase[None, :, :, None] * U * half_phase[None, :, None, :]
        S = U[:, 2] @ U[:, 1] @ U[:, 0]
        nch = -(-n // m)
        pad = nch * m - n
        if pad:
            S = np.concatenate([S, np.broadcast_to(eye, (pad, D, D))])
        A = S.reshape(nch, m, D, D)
        while A.shape[1] > 1:
            if A.shape[1] % 2:
                A = np.concatenate([A, np.broadcast_to(eye, (nch, 1, D, D))], axis=1)
            A = A[:, 1::2] @ A[:, 0::2]
        A = A[:, 0]
        for k in range(nch):
            h = (m if k < nch - 1 or not pad else m - pad) * dt
            E = _dissipator_exp(sim, h / 2)
            r = E @ r
            x = r.T.reshape(K, D, D)
            x = A[k] @ x @ A[k].conj().T
            r = x.reshape(K, D * D).T
            r = E @ r
    r = r.T.reshape(K, D, D)
    return r if batch else r[0]

def _fast_idle(sim, duration, rho):
    from scipy.sparse.linalg import expm_multiply
    Ld = _static_superop(sim)[3]
    D = sim.D
    K = rho.shape[0] if rho.ndim == 3 else 1
    r = expm_multiply(Ld * duration, rho.reshape(K, D * D).T).T.reshape(rho.shape)
    energy = np.diag(sim.H_static.full()).real
    phase = np.exp(-1j * energy * duration)
    return phase[:, None] * r * phase.conj()[None, :]

def run(sim, sequence, rho0=None, t0=0.0, return_state=False, *, _prefix_cache=None, _stats=None):
    from .core import Segment
    if apparatus_active(sim):
        return _run_controlled(sim, sequence, rho0, t0, return_state)
    state = sim.ground() if rho0 is None else rho0
    clock = float(t0)
    prefix = sha256(repr(t0).encode()).digest()
    for element in sequence:
        duration = element.duration if isinstance(element, Segment) else float(element)
        if duration <= 0:
            continue
        if _prefix_cache is not None:
            prefix = sha256(prefix + repr((clock, element)).encode()).digest()
            if prefix in _prefix_cache:
                state = _prefix_cache[prefix].copy()
                _prefix_cache.move_to_end(prefix)
                _stats['prefix_hits'] += 1
                clock += duration
                continue
        driven = isinstance(element, Segment) and any((p.amp for p in element.pulses))
        state = _evolve(sim, element.pulses if driven else [], clock, duration, state, driven)
        if _stats is not None:
            _stats['interval_solves'] += 1
        if _prefix_cache is not None:
            _prefix_cache[prefix] = state.copy()
            limit = max(1, 64 * 1024 ** 2 // (state.shape[0] * state.shape[1] * 16))
            if len(_prefix_cache) > limit:
                _prefix_cache.popitem(last=False)
        clock += duration
    if state.issuper:
        return state
    pops = sim.populations(state)
    return (pops, state, clock) if return_state else pops

def run_many(sim, sequences, rho0=None, t0=0.0, return_state=False):
    cache = OrderedDict()
    stats = {'prefix_hits': 0, 'interval_solves': 0}
    use_cache = not apparatus_active(sim)
    try:
        for sequence in sequences:
            yield run(sim, sequence, rho0, t0, return_state, _prefix_cache=cache if use_cache else None, _stats=stats)
    finally:
        sim.last_cosine_batch_stats = stats
        cache.clear()

def _evolve(sim, pulses, clock, duration, state, driven):
    if FAST_PROPAGATION and (not state.issuper):
        values = state.full()
        values = _fast_driven(sim, pulses, clock, duration, values) if driven else _fast_idle(sim, duration, values)
        state = qt.Qobj(values, dims=state.dims)
    elif FAST_PROPAGATION:
        cols = state.full()
        D = sim.D
        batch = np.stack([cols[:, k].reshape(D, D, order='F') for k in range(cols.shape[1])])
        batch = _fast_driven(sim, pulses, clock, duration, batch) if driven else _fast_idle(sim, duration, batch)
        state = qt.Qobj(np.column_stack([m.reshape(-1, order='F') for m in batch]), dims=state.dims, superrep='super')
    elif not driven:
        energy = np.diag(sim.H_static.full()).real
        result = qt.mesolve(0 * sim.H_static, state, [0.0, duration], sim.c_ops, options={**sim.opts, 'normalize_output': False}).final_state
        r = np.exp(-1j * energy * duration)
        values = result.full()
        if state.issuper:
            values = np.kron(r.conj(), r)[:, None] * values
        else:
            values = r[:, None] * values * r.conj()[None, :]
        state = qt.Qobj(values, dims=state.dims, superrep='super' if state.issuper else None)
    else:
        opts = {**sim.opts, 'normalize_output': False, 'max_step': min(0.02, duration / 20)}
        state = qt.mesolve(drive_hamiltonian(sim, pulses, clock), state, [clock, clock + duration], sim.c_ops, options=opts).final_state
    return state

def _run_controlled(sim, sequence, rho0, t0, return_state):
    commands, total, head = controlled_schedule(sim, sequence)
    boundaries = {0.0, total}
    for pulse, _, start, _ in commands:
        boundaries.update((start, start + pulse.duration))
        if pulse.shape == 'gaussian_square':
            rise = min(pulse.rise, pulse.duration / 2)
            boundaries.update((start + rise, start + pulse.duration - rise))
    boundaries = sorted((t for t in boundaries if 0 <= t <= total))
    state = sim.ground() if rho0 is None else rho0
    for a, b in zip(boundaries, boundaries[1:]):
        active = [item for item in commands if item[2] < b and item[2] + item[0].duration > a]
        waveform = SourceWaveform(sim, active, head, t0)
        state = _evolve(sim, waveform, t0 + a, b - a, state, waveform.bound > 0)
    if state.issuper:
        return state
    pops = sim.populations(state)
    return (pops, state, t0 + total) if return_state else pops
