from dataclasses import replace
from collections import OrderedDict
from hashlib import sha256
from itertools import product
import cmath
import numpy as np
import qutip as qt
from .numerics import expm, apply_propagator
from scipy.special import roots_hermitenorm
from .core import Pulse, Segment, TWO_PI
from .hardware import noise_factor

def iq_matrix(gain, skew):
    return np.array([[1.0, gain * np.sin(skew)], [0.0, gain * np.cos(skew)]])

def sidebands(matrix):
    a, b, c, d = matrix.ravel()
    return ((a + d + 1j * (c - b)) / 2, (a - d + 1j * (c + b)) / 2)

def schedule(sim, sequence):
    controls = getattr(sim, 'hardware_controls', {})
    delays = sim.dev.physics.get('delays', [0.0] * sim.n)
    commands, clock = ([], 0.0)
    for el in sequence:
        if isinstance(el, Segment):
            if type(el) is not Segment:
                raise ValueError('Optional pair hardware does not support coupler segments')
            for p in el.pulses:
                if p.duration < 0 or not np.isfinite(p.duration) or (not np.isfinite(p.start)):
                    raise ValueError('Invalid pulse timing')
                weights = [complex(*z) for z in p.source_weights] if p.source_weights is not None else [complex(k == p.line) for k in range(sim.n)]
                if len(weights) != sim.n:
                    raise ValueError('Wrong number of physical source weights')
                for k, weight in enumerate(weights):
                    if not weight or not p.amp or (not p.duration):
                        continue
                    delay = delays[k] + controls.get('channel_delay', {}).get(k, 0.0)
                    commands.append((replace(p, line=k, source_weights=None), weight, clock + p.start + delay, delay))
            clock += el.duration
        else:
            if not np.isfinite(el) or el < 0:
                raise ValueError('Invalid wait duration')
            clock += float(el)
    head = -min([0.0] + [start for _, _, start, _ in commands])
    end = max([clock] + [start + p.duration for p, _, start, _ in commands])
    return ([(p, w, start + head, delay) for p, w, start, delay in commands], end + head, head)

def duration(sim, sequence):
    return schedule(sim, sequence)[1]

def source_signals(sim, commands, head, t0, wref):
    signals = []
    controls = getattr(sim, 'hardware_controls', {})
    mixers = sim.dev.physics.get('mixers')
    betas = sim.dev.physics.get('amplitude_compression', [0.0] * sim.n)
    for line in range(sim.n):
        beta = betas[line]
        source = [item for item in commands if item[0].line == line]
        main, image, leak = (1.0 + 0j, 0j, 0j)
        if mixers is not None:
            m = mixers[line]
            transfer = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0)) @ iq_matrix(controls.get('mixer_gain', {}).get(line, 1.0), controls.get('mixer_skew', {}).get(line, 0.0))
            main, image = sidebands(transfer)
            dc = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0)) @ np.array([controls.get('mixer_dc_i', {}).get(line, 0.0), controls.get('mixer_dc_q', {}).get(line, 0.0)])
            leak = complex(*m.get('leakage', [0.0, 0.0])) + complex(*dc)
        if not source and (not leak):
            continue
        delay = sim.dev.physics.get('delays', [0.0] * sim.n)[line] + controls.get('channel_delay', {}).get(line, 0.0)
        line_rf_delay = delay if sim.dev.physics.get('delay_kind', 'envelope') == 'rf' else 0.0
        carrier = controls.get('lo_freq', {}).get(line, 5.0) if mixers is not None else source[0][0].freq
        hardware_phase = sim.dev.physics.get('ampm', [0.0] * sim.n)[line]
        predistortion = controls.get('ampm_compensation', {}).get(line, 0.0)
        if mixers is None:
            hardware_phase, predistortion = (hardware_phase + predistortion, 0.0)
        global_phase = -TWO_PI * carrier * (head + line_rf_delay) if mixers is not None else 0.0

        def make(source=source, carrier=carrier, beta=beta, hardware_phase=hardware_phase, predistortion=predistortion, main=main, image=image, leak=leak, global_phase=global_phase, mixer=mixers is not None):
            last_t, last_z = (None, None)

            def value(t):
                nonlocal last_t, last_z
                if t != last_t:
                    z = 0j
                    for pulse, weight, start, delay in source:
                        e, de = pulse.env_scalar(t - t0 - start)
                        if pulse.drag and (not np.isfinite(pulse.drag_anharm) or not pulse.drag_anharm):
                            raise ValueError('Invalid DRAG anharmonicity')
                        env = complex(e, pulse.drag * de / (TWO_PI * pulse.drag_anharm) if pulse.drag else 0.0)
                        rf_delay = delay if sim.dev.physics.get('delay_kind', 'envelope') == 'rf' else 0.0
                        if mixer:
                            z += pulse.amp * weight * env * cmath.exp(1j * (pulse.phase + TWO_PI * (pulse.freq - carrier) * (t - head - rf_delay)))
                        else:
                            z += pulse.amp * weight * env * cmath.exp(1j * (pulse.phase + TWO_PI * ((pulse.freq - carrier) * t - pulse.freq * (head + rf_delay))))
                    if predistortion:
                        z *= cmath.exp(1j * predistortion * abs(z) ** 2)
                    if mixer:
                        if image:
                            if not 0.1 <= 2 * carrier - max([p.freq for p, _, _, _ in source] or [carrier]) <= 15:
                                raise ValueError('Mixer sideband outside modeled RF range')
                        z = main * z + image * z.conjugate() + leak
                    power = abs(z) ** 2
                    last_z = z / np.sqrt(1 + beta * power) * cmath.exp(1j * (hardware_phase * power + (TWO_PI * carrier - wref) * t + global_phase))
                    last_t = t
                return last_z
            return (lambda t: value(t).real, lambda t: value(t).imag)
        real, imag = make()
        real.carrier = imag.carrier = carrier
        signals.append((line, real, imag))
    return signals

def _compression_terms(sim, commands, head, t0, wref):
    c = sim.dev.crosstalk_matrix()
    terms = []
    for line, real, imag in source_signals(sim, commands, head, t0, wref):
        gains = np.pi * sim.dev.qubits[line].drive_scale * c[:, line]
        a = sum((v.real * sim.X[j] + v.imag * sim.Y[j] for j, v in enumerate(gains)), 0 * sim.X[0])
        b = sum((-v.imag * sim.X[j] + v.real * sim.Y[j] for j, v in enumerate(gains)), 0 * sim.X[0])
        terms.extend([[a, real], [b, imag]])
    return terms

def _reference_terms(sim, commands, head, t0, wref, *, include_lo=True):
    if {'amplitude_compression', 'ampm'} & sim.dev.physics.keys():
        return _compression_terms(sim, commands, head, t0, wref)
    c = sim.dev.crosstalk_matrix()
    controls = getattr(sim, 'hardware_controls', {})
    mixers = sim.dev.physics.get('mixers')
    terms = []
    for p, weight, start, delay in commands:
        if p.drag and (not np.isfinite(p.drag_anharm) or not p.drag_anharm):
            raise ValueError('Nonzero DRAG needs finite nonzero source anharmonicity')
        lo = controls.get('lo_freq', {}).get(p.line, 5.0)
        main, image = (1.0 + 0j, 0j)
        if mixers is not None:
            m = mixers[p.line]
            transfer = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0)) @ iq_matrix(controls.get('mixer_gain', {}).get(p.line, 1.0), controls.get('mixer_skew', {}).get(p.line, 0.0))
            main, image = sidebands(transfer)
        rf_delay = delay if sim.dev.physics.get('delay_kind', 'envelope') == 'rf' else 0.0
        for mirror, coefficient, freq in ((False, main, p.freq), (True, image, 2 * lo - p.freq)):
            if not coefficient:
                continue
            if not 0.1 <= freq <= 15:
                raise ValueError('Mixer sideband outside modeled RF range')
            for j in range(sim.n):
                coupling = TWO_PI * sim.dev.qubits[p.line].drive_scale * p.amp * c[j, p.line] * coefficient / 2
                if not coupling:
                    continue

                def z(t, p=p, weight=weight, start=start, mirror=mirror, freq=freq, coupling=coupling, rf_delay=rf_delay):
                    e, de = p.env_scalar(t - t0 - start)
                    env = weight * complex(e, p.drag * de / (TWO_PI * p.drag_anharm) if p.drag else 0.0)
                    env *= cmath.exp(1j * p.phase)
                    if mirror:
                        env = env.conjugate()
                    return coupling * env * cmath.exp(1j * ((TWO_PI * freq - wref) * t - TWO_PI * freq * (head + rf_delay)))
                terms.extend([[sim.X[j], lambda t, z=z: z(t).real], [sim.Y[j], lambda t, z=z: z(t).imag]])
    if mixers is not None and include_lo:
        for k, m in enumerate(mixers):
            dc = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0)) @ np.array([controls.get('mixer_dc_i', {}).get(k, 0.0), controls.get('mixer_dc_q', {}).get(k, 0.0)])
            leak = complex(*m.get('leakage', [0.0, 0.0])) + complex(*dc)
            lo = controls.get('lo_freq', {}).get(k, 5.0)
            delay = sim.dev.physics.get('delays', [0.0] * sim.n)[k] + controls.get('channel_delay', {}).get(k, 0.0)
            rf_delay = delay if sim.dev.physics.get('delay_kind', 'envelope') == 'rf' else 0.0
            for j in range(sim.n):
                coupling = TWO_PI * sim.dev.qubits[k].drive_scale * c[j, k] * leak / 2
                if not coupling:
                    continue

                def z(t, coupling=coupling, lo=lo, rf_delay=rf_delay):
                    return coupling * cmath.exp(1j * ((TWO_PI * lo - wref) * t - TWO_PI * lo * (head + rf_delay)))
                terms.extend([[sim.X[j], lambda t, z=z: z(t).real], [sim.Y[j], lambda t, z=z: z(t).imag]])
    return terms

def _terms(sim, commands, head, t0, wref, *, include_lo=True):
    if {'amplitude_compression', 'ampm'} & sim.dev.physics.keys():
        return _compression_terms(sim, commands, head, t0, wref)
    c = sim.dev.crosstalk_matrix()
    controls = getattr(sim, 'hardware_controls', {})
    mixers = sim.dev.physics.get('mixers')
    groups = {}

    def add(key, p, start, beta, frequency, gains):
        if key not in groups:
            groups[key] = [p, start, beta, frequency, np.zeros(sim.n, complex)]
        groups[key][4] += gains
    for p, weight, start, delay in commands:
        if p.drag and (not np.isfinite(p.drag_anharm) or not p.drag_anharm):
            raise ValueError('Nonzero DRAG needs finite nonzero source anharmonicity')
        lo = controls.get('lo_freq', {}).get(p.line, 5.0)
        main, mirror_gain = (1.0 + 0j, 0j)
        if mixers is not None:
            m = mixers[p.line]
            transfer = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0)) @ iq_matrix(controls.get('mixer_gain', {}).get(p.line, 1.0), controls.get('mixer_skew', {}).get(p.line, 0.0))
            main, mirror_gain = sidebands(transfer)
        rf_delay = delay if sim.dev.physics.get('delay_kind', 'envelope') == 'rf' else 0.0
        for mirror, coefficient, freq in ((False, main, p.freq), (True, mirror_gain, 2 * lo - p.freq)):
            if not coefficient:
                continue
            if not 0.1 <= freq <= 15:
                raise ValueError('Mixer sideband outside modeled RF range')
            beta = p.drag / (TWO_PI * p.drag_anharm) if p.drag else 0.0
            gain = weight * cmath.exp(1j * p.phase)
            if mirror:
                beta = -beta
                gain = gain.conjugate()
            gain *= np.pi * sim.dev.qubits[p.line].drive_scale * p.amp * coefficient
            gain *= cmath.exp(-1j * TWO_PI * freq * (head + rf_delay))
            key = (p.shape, p.duration, p.rise, p.n_sigma, p.smooth_edges, start, beta, freq)
            add(key, p, start, beta, freq, gain * c[:, p.line])
    if mixers is not None and include_lo:
        for k, m in enumerate(mixers):
            dc = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0)) @ np.array([controls.get('mixer_dc_i', {}).get(k, 0.0), controls.get('mixer_dc_q', {}).get(k, 0.0)])
            leak = complex(*m.get('leakage', [0.0, 0.0])) + complex(*dc)
            if not leak:
                continue
            freq = controls.get('lo_freq', {}).get(k, 5.0)
            delay = sim.dev.physics.get('delays', [0.0] * sim.n)[k] + controls.get('channel_delay', {}).get(k, 0.0)
            rf_delay = delay if sim.dev.physics.get('delay_kind', 'envelope') == 'rf' else 0.0
            gain = np.pi * sim.dev.qubits[k].drive_scale * leak * cmath.exp(-1j * TWO_PI * freq * (head + rf_delay))
            add(('lo', freq), None, 0.0, 0.0, freq, gain * c[:, k])
    terms = []
    for p, start, beta, freq, gains in groups.values():
        if not np.any(gains):
            continue
        a, b = (0 * sim.X[0], 0 * sim.X[0])
        for j, gain in enumerate(gains):
            a += gain.real * sim.X[j] + gain.imag * sim.Y[j]
            b += -gain.imag * sim.X[j] + gain.real * sim.Y[j]

        def make(p=p, start=start, beta=beta, dw=TWO_PI * freq - wref):
            last_t, last_z = (None, None)

            def value(t):
                nonlocal last_t, last_z
                if t != last_t:
                    if p is None:
                        envelope = 1.0 + 0j
                    else:
                        e, de = p.env_scalar(t - t0 - start)
                        envelope = complex(e, beta * de)
                    last_t, last_z = (t, envelope * cmath.exp(1j * dw * t))
                return last_z
            return (lambda t: value(t).real, lambda t: value(t).imag)
        real, imag = make()
        real.carrier = imag.carrier = freq
        terms.extend([[a, real], [b, imag]])
    return terms

def _constant_interval(sim, active, a, b, head, t0, wref):
    for p, _, start, _ in active:
        if p.shape == 'square':
            continue
        if p.shape != 'gaussian_square' or a < start + p.rise or b > start + p.duration - p.rise:
            return None
    if {'amplitude_compression', 'ampm'} & sim.dev.physics.keys() and (len({p.freq for p, _, _, _ in active}) > 1 or 'mixers' in sim.dev.physics):
        return None
    terms = _terms(sim, active, head, t0, wref)
    carriers = {f.carrier for _, f in terms}
    if len(carriers) > 1:
        return None
    frame = TWO_PI * next(iter(carriers)) if carriers else wref
    local = _terms(sim, active, head, t0, frame)
    h = sim._static_frame_H(frame)
    for op, coefficient in local:
        h = h + op * coefficient(t0 + (a + b) / 2)
    return (h, frame)

def _interval_commands(active, end):
    result = []
    for p, weight, start, delay in active:
        if p.shape == 'square':
            p = replace(p, duration=end - start)
        elif p.shape == 'gaussian_square' and end <= start + p.duration - p.rise:
            p = replace(p, duration=max(2 * p.rise, end - start + p.rise))
        result.append((p, weight, start, delay))
    return result

def _flat_commands(active, a, b):
    result = []
    for p, weight, start, delay in active:
        if p.shape != 'square' and (p.shape != 'gaussian_square' or a < start + p.rise or b > start + p.duration - p.rise):
            return None
        result.append((replace(p, shape='square', duration=0.0, rise=0.0, drag=0.0), weight, start, delay))
    return result

def _rotate(sim, state, time, frequency, *, inverse=False):
    if not state.issuper:
        fn = sim.to_lab if inverse else sim.to_frame
        return fn(state, time, frequency)
    rotation = sim._R(time, -frequency if inverse else frequency)
    values = np.kron(rotation.conj(), rotation)[:, None] * state.full()
    return qt.Qobj(values, dims=state.dims, superrep='super')

def _exponential_action(sim, h, duration, state):
    generator = qt.liouvillian(h, sim.c_ops).full()
    key = (duration, sha256(generator.tobytes()).digest())
    cache = getattr(sim, '_extended_propagators', None)
    if cache is None:
        cache = sim._extended_propagators = OrderedDict()
    propagator = cache.get(key)
    if propagator is None:
        propagator = expm(generator * duration)
        limit = min(128, 32 * 1024 ** 2 // propagator.nbytes)
        if limit:
            cache[key] = propagator
            while len(cache) > limit:
                cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    values = state.full() if state.issuper else state.full().reshape(-1, order='F')
    vector = np.einsum('ij,j...->i...', propagator, values)
    if not np.isfinite(vector).all():
        raise FloatingPointError('Nonfinite exact propagation')
    if state.issuper:
        return qt.Qobj(vector, dims=state.dims, superrep='super')
    return qt.Qobj(vector.reshape((sim.D, sim.D), order='F'), dims=state.dims)
BATH_IDLE_EXACT = True
BATH_IDLE_STEP = 0.5

def _small_expm(G):
    w, V = np.linalg.eig(G)
    if np.linalg.cond(V) < 1000000.0:
        return V * np.exp(w) @ np.linalg.inv(V)
    return expm(G)

def _bath_idle(sim, h, bath_h, collapse_ops, state, start, end):
    static_c = [c for c in collapse_ops if not isinstance(c, list)]
    dynamic_c = [c for c in collapse_ops if isinstance(c, list)]
    edges = [start]
    step = BATH_IDLE_STEP
    while edges[-1] < end:
        edges.append(min(end, edges[-1] + step))
        step = min(step * 1.1, 10 * BATH_IDLE_STEP)
    base = qt.liouvillian(h, static_c).full()
    ops_h = [(qt.liouvillian(op).full(), f) for op, f in bath_h]
    ops_c = [(qt.lindblad_dissipator(op).full(), f) for op, f in dynamic_c]
    vector = state.full() if state.issuper else state.full().reshape(-1, order='F')
    for a, b in zip(edges[:-1], edges[1:]):
        m = 0.5 * (a + b)
        generator = base.copy()
        for L, f in ops_h:
            generator = generator + f(m) * L
        for L, f in ops_c:
            generator = generator + abs(f(m)) ** 2 * L
        vector = _small_expm(generator * (b - a)) @ vector
    if state.issuper:
        return qt.Qobj(vector, dims=state.dims, superrep='super')
    return qt.Qobj(vector.reshape((sim.D, sim.D), order='F'), dims=state.dims)

def run(sim, sequence, rho0=None, t0=0.0, return_state=False, *, _prefix_cache=None, _plateau_cache=None, _stats=None, _active_terms=True, _strict_prefix=False):
    if 'cosine_transmon' in sim.dev.physics:
        from .cosine_transmon import run as run_cosine
        return run_cosine(sim, sequence, rho0, t0, return_state)
    commands, total, head = schedule(sim, sequence)
    from .driven_baths import terms as bath_terms, enabled as bath_enabled
    bath_h, collapse_ops = bath_terms(sim, commands, total, head, t0)
    dynamic_bath = bath_enabled(sim.dev)
    initial = sim.ground() if rho0 is None else rho0
    wref = sim.w_frame[0]
    rho_frame = _rotate(sim, initial, t0, wref)
    terms = None if _active_terms else _reference_terms(sim, commands, head, t0, wref)
    chain = bool({'amplitude_compression', 'ampm'} & sim.dev.physics.keys())
    lo_terms = _terms(sim, [], head, t0, wref) if _active_terms and (not chain) else []
    controls = getattr(sim, 'hardware_controls', {})
    continuous_drive = False
    for k, mixer in enumerate(sim.dev.physics.get('mixers', [])):
        dc = iq_matrix(mixer.get('gain', 1.0), mixer.get('skew', 0.0)) @ np.array([controls.get('mixer_dc_i', {}).get(k, 0.0), controls.get('mixer_dc_q', {}).get(k, 0.0)])
        leakage = complex(*mixer.get('leakage', [0.0, 0.0])) + complex(*dc)
        continuous_drive |= leakage != 0
    boundaries = {0.0, total}
    for p, _, start, _ in commands:
        boundaries.update((start, start + p.duration))
        if p.shape == 'gaussian_square':
            boundaries.update((start + min(p.rise, p.duration / 2), start + p.duration - min(p.rise, p.duration / 2)))
    boundaries = sorted((b for b in boundaries if 0 <= b <= total))
    intervals = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            continue
        active = [command for command in commands if command[2] < b and command[2] + command[0].duration > a]
        features = [p.duration / p.n_sigma if p.shape == 'gaussian' else p.rise if p.shape == 'gaussian_square' else p.duration for p, _, _, _ in active]
        if not _active_terms:
            local_terms = terms
        elif chain:
            local_terms = _terms(sim, active, head, t0, wref)
        else:
            local_terms = _terms(sim, active, head, t0, wref, include_lo=False) + lo_terms
        signature = repr((a, b, active if _strict_prefix else _interval_commands(active, b))).encode() if _prefix_cache is not None else b''
        if _strict_prefix and dynamic_bath:
            bath_edges = sorted({0.0, total} | {max(0.0, min(total, x)) for p, _, start, _ in commands for x in (start, start + p.duration, start + p.rise, start + p.duration - p.rise)})
            carriers = [next((p.freq for p, _, _, _ in commands if p.line == k), sim.dev.qubits[k].freq) for k in range(sim.n)]
            signature += repr(([x for x in bath_edges if x <= b], carriers, [c for c in commands if c[2] <= b])).encode()
        flat = _flat_commands(active, a, b) if _plateau_cache is not None else None
        plateau = repr((a, flat)).encode() if flat is not None else None
        constant = _constant_interval(sim, active, a, b, head, t0, wref) if _active_terms and (not dynamic_bath) and sim.exact_idle and (sim.exact_flat or not active) else None
        intervals.append((a, b, features, local_terms, signature, constant, plateau))
    static = sim._static_frame_H(wref)

    def evolve(offset):
        h = static
        for k, x in enumerate(offset):
            if x:
                h = h + TWO_PI * x * sim.num[k]
        state = rho_frame
        prefix = sha256(repr((t0, head, offset.tolist())).encode()).digest()
        for a, b, features, local_terms, signature, constant, plateau in intervals:
            plateau_key = sha256(prefix + plateau).digest() if plateau is not None else None
            if _prefix_cache is not None:
                prefix = sha256(prefix + signature).digest()
                cached = _prefix_cache.get(prefix)
                if cached is not None:
                    state = cached.copy()
                    _prefix_cache.move_to_end(prefix)
                    if _stats is not None:
                        _stats['prefix_hits'] += 1
                    continue
            checkpoint = _plateau_cache.get(plateau_key) if plateau_key is not None else None
            if checkpoint is not None and a < checkpoint[0] <= b:
                previous_end, state = checkpoint
                state = state.copy()
                if _stats is not None:
                    _stats['plateau_hits'] += 1
                    _stats['reused_time_ns'] += previous_end - a
                a = previous_end
            if _stats is not None:
                _stats['interval_solves'] += 1
            opts = sim._qutip_segment_options(b - a)
            opts['normalize_output'] = False
            if b == a:
                pass
            elif constant is not None:
                hc, frame = constant
                for k, x in enumerate(offset):
                    if x:
                        hc = hc + TWO_PI * x * sim.num[k]
                shifted = _rotate(sim, state, t0 + a, frame - wref)
                shifted = _exponential_action(sim, hc, b - a, shifted)
                state = _rotate(sim, shifted, t0 + b, frame - wref, inverse=True)
            elif sim.exact_idle and (not features) and (not continuous_drive) and (not dynamic_bath):
                generator = qt.liouvillian(h, sim.c_ops).full()
                vector = apply_propagator(expm(generator * (b - a)), state.full().reshape(-1, order='F'))
                state = qt.Qobj(vector.reshape((sim.D, sim.D), order='F'), dims=state.dims)
            elif BATH_IDLE_EXACT and sim.exact_idle and (not features) and (not continuous_drive) and dynamic_bath and (not local_terms):
                state = _bath_idle(sim, h, bath_h, collapse_ops, state, t0 + a, t0 + b)
            else:
                if features:
                    step = max(0.0001, min(features) / 4)
                    opts['max_step'] = min(opts.get('max_step', 0) or step, step)
                state = qt.mesolve([h] + local_terms + bath_h, state, [t0 + a, t0 + b], c_ops=collapse_ops, options=opts).final_state
            if plateau_key is not None and (checkpoint is None or b > checkpoint[0]):
                _plateau_cache[plateau_key] = (b, state.copy())
                _plateau_cache.move_to_end(plateau_key)
                if len(_plateau_cache) > 512:
                    _plateau_cache.popitem(last=False)
            if _prefix_cache is not None:
                _prefix_cache[prefix] = state.copy()
                if len(_prefix_cache) > 512:
                    _prefix_cache.popitem(last=False)
        return state.full()
    factor = noise_factor(sim.dev, getattr(sim, 'hardware_controls', {}))
    rank = factor.shape[1]
    if not rank or not total:
        out, error, order = (evolve(np.zeros(sim.n)), 0.0, 1)
    else:
        previous, error, order = (None, np.inf, sim.noise_order)
        while True:
            nodes, weights = roots_hermitenorm(order)
            weights = weights / np.sqrt(2 * np.pi)
            out = np.zeros(initial.shape, complex)
            for indices in product(range(order), repeat=rank):
                weight = float(np.prod(weights[list(indices)]))
                if weight < 1e-16:
                    continue
                out += weight * evolve(factor @ nodes[list(indices)])
            if previous is not None:
                difference = out - previous
                error = max((float(np.linalg.norm(difference[:, k].reshape(sim.D, sim.D, order='F'), ord='nuc')) for k in range(difference.shape[1]))) if initial.issuper else float(np.linalg.norm(difference, ord='nuc'))
                if error <= sim.noise_tolerance:
                    break
            if order >= sim.noise_max_order:
                raise ValueError(f'Slow-noise quadrature did not converge (order={order}, error={error:.3g}); increase noise_max_order')
            previous = out
            order = min(2 * order - 1, sim.noise_max_order)
    sim.last_noise_convergence = {'order': order, 'state_trace_norm_change': error}
    state = _rotate(sim, qt.Qobj(out, dims=initial.dims, superrep='super' if initial.issuper else None), t0 + total, wref, inverse=True)
    if initial.issuper:
        return state
    pops = sim.populations(state)
    return (pops, state, t0 + total) if return_state else pops

def run_many(sim, sequences, rho0=None, t0=0.0, return_state=False):
    if 'cosine_transmon' in sim.dev.physics:
        from .cosine_transmon import run_many as run_cosine_many
        yield from run_cosine_many(sim, sequences, rho0, t0, return_state)
        return
    cache = OrderedDict()
    plateaus = OrderedDict()
    stats = {'prefix_hits': 0, 'interval_solves': 0, 'plateau_hits': 0, 'reused_time_ns': 0.0}
    try:
        for sequence in sequences:
            yield run(sim, sequence, rho0, t0, return_state, _prefix_cache=cache, _plateau_cache=plateaus, _stats=stats)
    finally:
        sim.last_extended_batch_stats = stats.copy()
        cache.clear()
        plateaus.clear()

def run_states(sim, sequence, initials, t0=0.0):
    initials = list(initials)
    side = int(np.sqrt(len(initials)))
    if sim.n > 2 or not side or side * side != len(initials):
        raise ValueError('Single/pair state batch requires a square number of states')
    columns = np.column_stack([rho.full().reshape(-1, order='F') for rho in initials])
    initial = qt.Qobj(columns, dims=[[sim.dims, sim.dims], [[side], [side]]], superrep='super')
    values = run(sim, sequence, rho0=initial, t0=t0).full()
    return [qt.Qobj(values[:, k].reshape(sim.D, sim.D, order='F'), dims=[sim.dims] * 2) for k in range(len(initials))]
