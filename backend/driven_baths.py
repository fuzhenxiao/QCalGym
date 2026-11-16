import cmath
import numpy as np
from scipy.integrate import solve_ivp
from .scalar_dense import scalar_evaluator
KEYS = ('heating', 'parasitic_mode')

def enabled(dev):
    return any((key in dev.physics for key in KEYS))

def validate(dev):
    from .hardware import _number
    for key in KEYS:
        entries = dev.physics.get(key, [None] * dev.n)
        if not isinstance(entries, list) or len(entries) != dev.n:
            raise ValueError(f'{key} requires one entry per qubit')
        for p in entries:
            if p is None:
                continue
            if key == 'heating' and isinstance(p, dict) and (p.get('model') == 'electron_phonon'):
                from .thermal_noise import validate as validate_thermal
                validate_thermal(p)
                continue
            expected = {'eta', 'tau', 'rate', 'dephasing'} if key == 'heating' else {'freq', 'kappa', 'coupling', 'chi'}
            if not isinstance(p, dict) or set(p) != expected:
                raise ValueError(f'{key} requires {sorted(expected)}')
            for name, value in p.items():
                _number(value, f'{key}.{name}', None if name == 'chi' else 0.0)
            if key == 'heating' and p['tau'] <= 0:
                raise ValueError('Thermal recovery time must be positive')
            if key == 'parasitic_mode' and (p['kappa'] <= 0 or p['freq'] <= 0):
                raise ValueError('Mode frequency and linewidth must be positive')

def envelope(commands, line, t, carrier, head, t0):
    z = 0j
    for pulse, weight, start, delay in commands:
        if pulse.line != line:
            continue
        e, de = pulse.env_scalar(t - t0 - start)
        quadrature = pulse.drag * de / (2 * np.pi * pulse.drag_anharm) if pulse.drag else 0.0
        z += pulse.amp * weight * complex(e, quadrature) * cmath.exp(1j * (pulse.phase + 2 * np.pi * ((pulse.freq - carrier) * t - pulse.freq * head)))
    return z

def terms(sim, commands, total, head, t0):
    if not enabled(sim.dev) or total == 0:
        return ([], list(sim.c_ops))
    boundaries = sorted({0.0, total} | {max(0.0, min(total, x)) for p, _, start, _ in commands for x in (start, start + p.duration, start + p.rise, start + p.duration - p.rise)})
    hterms, collapses = ([], list(sim.c_ops))
    for key in KEYS:
        for k, params in enumerate(sim.dev.physics.get(key, [])):
            if params is None:
                continue
            carrier = next((p.freq for p, _, _, _ in commands if p.line == k), sim.dev.qubits[k].freq)

            def rhs(t, state, p=params, key=key, k=k, carrier=carrier):
                u = envelope(commands, k, t, carrier, head, t0)
                if key == 'heating' and isinstance(p, dict) and (p.get('model') == 'electron_phonon'):
                    from .thermal_noise import rhs as thermal_rhs
                    return [thermal_rhs(p, u, state[0])]
                if key == 'heating':
                    return [p['eta'] * abs(u) ** 2 - state[0] / p['tau']]
                return [-(np.pi * p['kappa'] + 2j * np.pi * (p['freq'] - carrier)) * state[0] - 2j * np.pi * p['coupling'] * u.conjugate()]
            state = np.zeros(1, complex if key == 'parasitic_mode' else float)
            if key == 'heating' and params.get('model') == 'electron_phonon':
                state[0] = params['base_temperature_K'] ** 2
            solutions = []
            for a, b in zip(boundaries, boundaries[1:]):
                if b <= a:
                    continue
                active = [p for p, _, start, _ in commands if p.line == k and start < b and (start + p.duration > a)]
                feature = min([p.duration / p.n_sigma if p.shape == 'gaussian' else max(0.01, p.rise) if p.shape == 'gaussian_square' else p.duration for p in active] or [b - a])
                result = solve_ivp(rhs, (t0 + a, t0 + b), state, method='DOP853', rtol=1e-09, atol=1e-12, max_step=max(0.0001, feature / 4), dense_output=True)
                if not result.success:
                    raise RuntimeError(result.message)
                state = result.y[:, -1]
                solutions.append(scalar_evaluator(result.sol))
            ends = np.array(boundaries[1:]) + t0
            last = [None, 0.0]

            def occupation(t, solutions=solutions, ends=ends, key=key, params=params, last=last):
                if t == last[0]:
                    return last[1]
                last[0] = t
                last[1] = _occupation(t, solutions, ends, key, params)
                return last[1]

            def _occupation(t, solutions, ends, key, params):
                index = min(int(np.searchsorted(ends, t)), len(solutions) - 1)
                value = solutions[index](t)
                if key == 'heating' and params.get('model') == 'electron_phonon':
                    from .thermal_noise import occupation as bose
                    return bose(np.sqrt(max(0.0, float(value))), params['noise_frequency_GHz'])
                return max(0.0, float(abs(value) ** 2 if key == 'parasitic_mode' else value))
            if key == 'heating':
                rate = params['rate']
                for op in (sim.a[k], sim.ad[k]):
                    collapses.append([op, lambda t, occupation=occupation, rate=rate: np.sqrt(rate * occupation(t))])
                dephasing = params['dephasing']
                collapses.append([sim.num[k], lambda t, occupation=occupation, dephasing=dephasing: np.sqrt(2 * dephasing * occupation(t))])
            else:
                hterms.append([2 * np.pi * params['chi'] * sim.num[k], occupation])
                rate = 8 * np.pi * params['chi'] ** 2 / params['kappa']
                collapses.append([sim.num[k], lambda t, occupation=occupation, rate=rate: np.sqrt(2 * rate * occupation(t))])
    return (hterms, collapses)
