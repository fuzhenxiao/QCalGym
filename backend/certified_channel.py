import numpy as np
import qutip as qt
from scipy.integrate import simpson
from .core import Segment

def score(sim, sequence, target, basis, frame):
    if len(sequence) != 1 or not isinstance(sequence[0], Segment):
        raise ValueError('Certified short-layer bounds require one pulse segment')
    segment = sequence[0]
    duration = segment.duration
    d = len(target)
    if not 0 < duration <= 50:
        raise ValueError('Certified short-layer bounds cover at most 50 ns')
    rate = sum((c.dag() * c for c in sim.c_ops), 0 * sim.H_static)
    gamma = float(np.linalg.eigvalsh(rate.full())[-1])
    w = 2 * np.pi * segment.pulses[0].freq
    static = sim._static_frame_H(w)
    drives = sim._drive_terms(segment, 0.0, w)
    forward = [static - 0.5j * rate] + drives
    backward = [static + 0.5j * rate] + drives
    final_basis = sim._R(duration, w)[:, None] * basis * frame.conj()[None, :]
    options = {**sim.opts, 'method': 'dop853', 'normalize_output': False, 'store_states': True, 'store_final_state': False}
    floor = max(5e-10, 50 * sim.opts['rtol'])
    previous = None
    for intervals in (32, 64, 128, 256, 512, 1024):
        times = np.linspace(0.0, duration, intervals + 1)
        options['max_step'] = duration / intervals
        v0 = qt.Qobj(basis, dims=[sim.dims, [d]])
        wt = qt.Qobj(final_basis, dims=[sim.dims, [d]])
        v = np.array([s.full() for s in qt.sesolve(forward, v0, times, options=options).states])
        states = qt.sesolve(backward, wt, times[::-1], options=options).states
        back = np.array([s.full() for s in states[::-1]])
        a0 = frame[:, None] * np.einsum('ka,kb->ab', basis.conj(), sim._R(duration, w).conj()[:, None] * v[-1])

        def contribution(a):
            norm = np.sum(abs(a) ** 2, axis=(-2, -1))
            overlap = abs(np.einsum('ab,...ab->...', target.conj(), a)) ** 2
            return ((norm + overlap) / (d * (d + 1)), norm / d)
        zero_f, zero_s = contribution(a0)
        density_f = np.zeros(len(times))
        density_s = np.zeros(len(times))
        for c in sim.c_ops:
            sparse = c.to('CSR').data.as_scipy()
            cv = (sparse @ v.transpose(1, 0, 2).reshape(sim.D, -1)).reshape(sim.D, len(times), d).transpose(1, 0, 2)
            a = np.einsum('tka,tkb->tab', back.conj(), cv, optimize=True)
            f, s = contribution(a)
            density_f += f
            density_s += s
        fidelity = float(zero_f + simpson(density_f, x=times))
        survival = float(zero_s + simpson(density_s, x=times))
        no_jump_survival = np.sum(abs(v) ** 2, axis=(1, 2)) / d
        first_mass = np.maximum(0.0, no_jump_survival[:-1] - no_jump_survival[1:])
        tail = float(np.sum(first_mass * -np.expm1(-gamma * (duration - times[:-1]))))
        if not np.isfinite([fidelity, survival, tail]).all() or np.max(no_jump_survival) > 1 + 1e-07:
            raise FloatingPointError('Invalid no-jump evolution')
        difference = max(abs(fidelity - previous[0]), abs(survival - previous[1])) if previous is not None else float('inf')
        if difference < 1e-09:
            numerical = 10 * difference + floor
            lower = max(0.0, fidelity - numerical)
            upper = min(1.0, fidelity + tail + numerical)
            return dict(fidelity=lower, fidelity_interval=[lower, upper], leakage=max(0.0, 1 - survival + numerical), leakage_interval=[max(0.0, 1 - survival - tail - numerical), max(0.0, 1 - survival + numerical)], input_states=d * d, computational_dimension=d, method='full_lindblad_zero_one_jump_with_omitted_jump_bound', value_kind='conservative_fidelity_lower_bound', omitted_jump_probability_bound=tail, numerical_allowance=numerical, quadrature_intervals=intervals)
        previous = (fidelity, survival)
    raise ValueError('Short-layer jump quadrature did not converge; full integration required')
