import math
import numpy as np
import qutip as qt
import scipy.linalg as la
from .numerics import expm, apply_propagator
from scipy.sparse.linalg import expm_multiply
from .core import Segment

def propagate(sim, sequence, initial):
    size = initial.shape[1]
    side = math.isqrt(size)
    if side * side != size:
        raise ValueError('Channel batch size must be square')
    dims = [[sim.dims, sim.dims], [[side], [side]]]
    data = initial
    t = 0.0
    calls = 0
    for element in sequence:
        pulse = isinstance(element, Segment)
        duration = element.duration if pulse else float(element)
        if not duration:
            continue
        w = 2 * np.pi * element.pulses[0].freq if pulse else sim.w_frame[0]
        if pulse:
            rotation = sim._R(t, w)
            data = np.kron(rotation.conj(), rotation)[:, None] * data
            h = [sim._static_frame_H(w)] + sim._drive_terms(element, t, w)
            features = [p.duration / p.n_sigma if p.shape == 'gaussian' else p.rise if p.shape == 'gaussian_square' else p.duration for p in element.pulses if p.duration > 0]
            opts = {**sim._qutip_segment_options(duration), 'normalize_output': False, 'max_step': min(features) / 4, 'store_states': False, 'store_final_state': True}

            def integrate(begin, end):
                nonlocal data, calls
                state = qt.Qobj(data, dims=dims, superrep='super')
                data = qt.mesolve(h, state, [begin, end], c_ops=sim.c_ops, options=opts).final_state.full()
                calls += 1
            if sim.exact_flat and sim._is_flat_top(element):
                rise = element.pulses[0].rise
                integrate(t, t + rise)
                hflat = sim._constant_H(element, w)
                generator = qt.liouvillian(hflat, sim.c_ops).to('CSR').data.as_scipy()
                plateau = duration - 2 * rise
                if sim.D <= 25:
                    key = ('operator_channel', sim._segment_key(element), element.pulses[0].phase)
                    transform = sim._full_flat_cache.get(key)
                    if transform is None:
                        transform = expm(generator.toarray() * plateau)
                        if len(sim._full_flat_cache) >= 4:
                            sim._full_flat_cache.clear()
                        sim._full_flat_cache[key] = transform
                    data = apply_propagator(transform, data)
                else:
                    data = expm_multiply(generator * plateau, data, traceA=generator.diagonal().sum() * plateau)
                integrate(t + duration - rise, t + duration)
            else:
                integrate(t, t + duration)
            rotation = sim._R(t + duration, w).conj()
        else:
            if sim.exact_idle:
                generator = qt.liouvillian(sim._static_frame_H(w), sim.c_ops).to('CSR').data.as_scipy()
                data = expm_multiply(generator * duration, data, traceA=generator.diagonal().sum() * duration)
            else:
                opts = {**sim._qutip_segment_options(duration), 'normalize_output': False, 'store_states': False, 'store_final_state': True}
                state = qt.Qobj(data, dims=dims, superrep='super')
                data = qt.mesolve(sim._static_frame_H(w), state, [0.0, duration], c_ops=sim.c_ops, options=opts).final_state.full()
                calls += 1
            rotation = sim._R(duration, w).conj()
        data = np.kron(rotation.conj(), rotation)[:, None] * data
        t += duration
    if not np.isfinite(data).all():
        raise FloatingPointError('Nonfinite full operator channel')
    return (data, calls)

def score(env, sequence, target, basis, frame, *, state_bytes=16 * 1024 ** 2):
    sim = env.sim
    d = len(target)
    batch = d * d
    while batch > 1 and sim.D ** 2 * batch * 16 > state_bytes:
        batch //= 4
    channel = np.empty((d * d, d * d), complex)
    calls = 0
    for start in range(0, d * d, batch):
        cols = np.arange(start, min(start + batch, d * d))
        i, j = (cols % d, cols // d)
        initial = np.einsum('ia,ja->ija', basis[:, i], basis[:, j].conj()).reshape(sim.D ** 2, len(cols), order='F')
        data, used = propagate(sim, sequence, initial)
        calls += used
        rho = data.reshape(sim.D, sim.D, len(cols), order='F')
        projected = np.einsum('ia,ijk,jb->abk', basis.conj(), rho, basis, optimize=True)
        projected = frame[:, None, None] * projected * frame.conj()[None, :, None]
        channel[:, cols] = projected.reshape(d * d, len(cols), order='F')
    diagonal = np.arange(d) * (d + 1)
    survival = float(channel[np.ix_(diagonal, diagonal)].sum().real)
    overlap = float(np.vdot(np.kron(target.conj(), target), channel).real)
    return {'fidelity': (overlap + survival) / (d * (d + 1)), 'leakage': max(0.0, 1 - survival / d), 'input_states': d * d, 'channel_method': 'full_lindblad_operator_batches', 'batch_columns': batch, 'ode_calls': calls}
