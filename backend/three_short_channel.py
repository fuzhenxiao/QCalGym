import numpy as np
import qutip as qt
from scipy.integrate import simpson
from .jump_channel import setup

class ConvergenceError(ValueError):
    pass

def score(sim, sequence, target, basis, frame):
    w, rate, static, pieces, duration = setup(sim, sequence)
    if not 0 < duration <= 600:
        raise ValueError('Short-operation bound requires 0 < duration <= 600 ns')
    d = len(target)
    gamma = float(np.linalg.eigvalsh(rate.full())[-1])
    terminal = sim._R(duration, w)[:, None] * basis * frame.conj()[None, :]
    floor = max(5e-10, 50 * sim.opts['rtol'])
    previous = None
    for intervals in (16, 32, 64, 128, 256, 512, 1024):
        grids = [np.linspace(begin, end, intervals + 1) for begin, end, _, _ in pieces]
        times = np.concatenate([grids[0]] + [g[1:] for g in grids[1:]])

        def integrate(backward=False):
            state = terminal if backward else basis
            chunks = []
            iterable = list(zip(pieces, grids))
            if backward:
                iterable.reverse()
            for (begin, end, _, terms), grid in iterable:
                h = [static + (0.5j if backward else -0.5j) * rate] + terms
                opts = {**sim.opts, 'method': 'dop853', 'normalize_output': False, 'store_states': True, 'store_final_state': False, 'max_step': (end - begin) / intervals}
                values = qt.sesolve(h, qt.Qobj(state, dims=[sim.dims, [d]]), grid[::-1] if backward else grid, options=opts).states
                block = np.array([s.full() for s in values])
                state = block[-1]
                chunks.append(block[::-1] if backward else block)
            if backward:
                chunks.reverse()
            return np.concatenate([chunks[0]] + [chunk[1:] for chunk in chunks[1:]])
        forward = integrate()
        back = integrate(True)
        a0 = frame[:, None] * np.einsum('ka,kb->ab', basis.conj(), sim._R(duration, w).conj()[:, None] * forward[-1])

        def contribution(a):
            norm = np.sum(abs(a) ** 2, axis=(-2, -1))
            overlap = abs(np.einsum('ab,...ab->...', target.conj(), a)) ** 2
            return ((norm + overlap) / (d * (d + 1)), norm / d)
        zero_f, zero_s = contribution(a0)
        density_f = np.zeros(len(times))
        density_s = np.zeros(len(times))
        for collapse in sim.c_ops:
            sparse = collapse.to('CSR').data.as_scipy()
            cv = (sparse @ forward.transpose(1, 0, 2).reshape(sim.D, -1)).reshape(sim.D, len(times), d).transpose(1, 0, 2)
            a = np.einsum('tka,tkb->tab', back.conj(), cv, optimize=True)
            f, s = contribution(a)
            density_f += f
            density_s += s
        integral_f = integral_s = 0.0
        for index, grid in enumerate(grids):
            part = slice(index * intervals, (index + 1) * intervals + 1)
            integral_f += simpson(density_f[part], x=grid)
            integral_s += simpson(density_s[part], x=grid)
        fidelity = float(zero_f + integral_f)
        survival = float(zero_s + integral_s)
        nojump = np.sum(abs(forward) ** 2, axis=(1, 2)) / d
        first_mass = np.maximum(0.0, nojump[:-1] - nojump[1:])
        tail = float(np.sum(first_mass * -np.expm1(-gamma * (duration - times[:-1]))))
        if not np.isfinite([fidelity, survival, tail]).all() or np.max(nojump) > 1 + 1e-07:
            raise FloatingPointError('Invalid no-jump evolution')
        difference = max(abs(fidelity - previous[0]), abs(survival - previous[1])) if previous else float('inf')
        if difference < 1e-09:
            numerical = 10 * difference + floor
            lower = max(0.0, fidelity - numerical)
            upper = min(1.0, fidelity + tail + numerical)
            return dict(fidelity=lower, fidelity_interval=[lower, upper], leakage=max(0.0, 1 - survival + numerical), leakage_interval=[max(0.0, 1 - survival - tail - numerical), max(0.0, 1 - survival + numerical)], input_states=d * d, computational_dimension=d, method='full_lindblad_composite_zero_one_jump_bound_v1', value_kind='conservative_fidelity_lower_bound', omitted_jump_probability_bound=tail, numerical_allowance=numerical, quadrature_intervals=intervals * len(pieces))
        previous = (fidelity, survival)
    raise ConvergenceError('Composite jump quadrature did not converge; full integration required')
