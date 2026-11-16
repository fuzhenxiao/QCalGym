import numpy as np
import qutip as qt
import scipy.linalg as la
import scipy.sparse as sp
from scipy.integrate import simpson
from .core import Segment

def setup(sim, sequence):
    pulse = next((s for s in sequence if isinstance(s, Segment) and s.pulses), None)
    w = 2 * np.pi * pulse.pulses[0].freq if pulse else sim.w_frame[0]
    rate = sum((c.dag() * c for c in sim.c_ops), 0 * sim.H_static)
    static = sim._static_frame_H(w)
    pieces = []
    t = 0.0
    for element in sequence:
        duration = element.duration if isinstance(element, Segment) else float(element)
        if duration:
            terms = sim._drive_terms(element, t, w) if isinstance(element, Segment) else []
            pieces.append((t, t + duration, element, terms))
        t += duration
    return (w, rate, static, pieces, t)

def fundamental_samples(sim, sequence, times, prepared):
    w, rate, static, pieces, total = prepared
    h0 = static - 0.5j * rate
    state = np.eye(sim.D, dtype=complex)
    yield state
    opts = {**sim.opts, 'method': 'dop853', 'normalize_output': False, 'store_states': True, 'store_final_state': False}

    def evolve(begin, end, h, constant=False):
        nonlocal state
        indices = np.flatnonzero((times > begin + 1e-10) & (times <= end + 1e-10))
        knots = list(times[indices])
        if not knots or abs(knots[-1] - end) > 1e-10:
            knots.append(end)
        if constant:
            eigen, v = la.eig(h.full())
            initial = la.solve(v, state)
        for offset in range(0, len(knots), 32):
            chunk = knots[offset:offset + 32]
            if constant:
                values = [v @ (np.exp(-1j * eigen * (t - begin))[:, None] * initial) for t in chunk]
            else:
                start = begin if offset == 0 else knots[offset - 1]
                result = qt.sesolve(h, qt.Qobj(state, dims=[sim.dims] * 2), [start] + chunk, options=opts)
                values = [s.full() for s in result.states[1:]]
            state = values[-1]
            for index, value in enumerate(values, offset):
                if index < len(indices):
                    yield value
    for begin, end, element, terms in pieces:
        if not isinstance(element, Segment):
            yield from evolve(begin, end, h0, True)
            continue
        opts['max_step'] = min((p.duration / p.n_sigma if p.shape == 'gaussian' else p.rise if p.shape == 'gaussian_square' else p.duration for p in element.pulses if p.duration > 0)) / 4
        h = [h0] + terms
        common = all((abs(2 * np.pi * p.freq - w) < 1e-12 for p in element.pulses))
        if common and sim._is_flat_top(element):
            rise = element.pulses[0].rise
            yield from evolve(begin, begin + rise, h)
            yield from evolve(begin + rise, end - rise, sim._constant_H(element, w) - 0.5j * rate, True)
            yield from evolve(end - rise, end, h)
        elif common and sim._is_constant(element):
            yield from evolve(begin, end, sim._constant_H(element, w) - 0.5j * rate, True)
        else:
            yield from evolve(begin, end, h)

def counted_effects(sim, basis, prepared):
    w, rate, static, pieces, total = prepared
    D = sim.D
    d = basis.shape[1]
    nojump = -1j * (qt.spre(static) - qt.spost(static)) - 0.5 * (qt.spre(rate) + qt.spost(rate))
    jump = sum((qt.sprepost(c, c.dag()) for c in sim.c_ops), 0 * nojump)
    A = nojump.dag().to('CSR').data.as_scipy()
    J = jump.dag().to('CSR').data.as_scipy()
    full = A + J
    generator = sp.block_diag([A, A, A, full], format='csr')
    ladder = sp.csr_matrix(([1.0, 1.0], ([1, 2], [0, 1])), shape=(4, 4))
    generator += sp.kron(ladder, J, format='csr')
    h = [qt.Qobj(1j * generator)]
    feature = total
    for begin, end, element, terms in pieces:
        if isinstance(element, Segment):
            feature = min(feature, *(p.duration / p.n_sigma if p.shape == 'gaussian' else p.rise if p.shape == 'gaussian_square' else p.duration for p in element.pulses if p.duration > 0))
        for op, coefficient in terms:
            delta = (-1j * (qt.spre(op) - qt.spost(op))).dag().to('CSR').data.as_scipy()
            h.append([qt.Qobj(1j * sp.kron(sp.eye(4), delta, format='csr')), lambda s, f=coefficient: f(total - s)])
    initial = np.zeros((4, D * D), complex)
    initial[0] = np.eye(D).reshape(-1, order='F')
    rotated = sim._R(total, w)[:, None] * basis
    initial[3] = (rotated @ rotated.conj().T).reshape(-1, order='F')
    opts = {**sim.opts, 'method': 'dop853', 'normalize_output': False, 'store_states': False, 'store_final_state': True, 'max_step': feature / 4, 'nsteps': max(sim.opts['nsteps'], int(total * 1000))}
    result = qt.sesolve(h, qt.Qobj(initial.reshape(-1)), [0.0, total], options=opts).final_state.full().reshape(4, D, D, order='C')
    effects = result.transpose(0, 2, 1)
    missing = np.eye(d) - basis.conj().T @ sum(effects[:3]) @ basis
    missing = (missing + missing.conj().T) / 2
    allowance = max(1e-09, 100 * sim.opts['rtol'])
    values = np.maximum(0.0, la.eigvalsh(missing) + allowance)
    tail = float((np.sqrt(values).sum() / d) ** 2 * d / (d + 1))
    survival = float(np.trace(basis.conj().T @ effects[3] @ basis).real / d)
    if not np.isfinite([tail, survival]).all() or not -allowance <= survival <= 1 + allowance:
        raise FloatingPointError('Invalid counted-adjoint result')
    return dict(tail_bound=tail, missing_probability_bound=float(values.mean()), survival=survival, allowance=allowance, missing_eigenvalues=values.tolist())

def score(sim, sequence, target, basis, frame, *, max_intervals=2048):
    prepared = setup(sim, sequence)
    w, rate, static, pieces, total = prepared
    d = len(target)
    effect = counted_effects(sim, basis, prepared)
    for endpoint in fundamental_samples(sim, sequence, np.array([0.0, total]), prepared):
        pass
    projection = frame[:, None] * (basis.conj().T * sim._R(total, w).conj()[None, :])
    final = target.conj().T @ projection @ endpoint
    previous = None
    for intervals in (16, 32, 64, 128, 256, 512, 1024, 2048):
        if intervals > max_intervals:
            break
        times = np.linspace(0.0, total, intervals + 1)
        channels = len(sim.c_ops)
        left = np.empty((len(times), channels, d * sim.D), complex)
        right = np.empty_like(left)
        one = np.zeros(len(times))
        for index, value in enumerate(fundamental_samples(sim, sequence, times, prepared)):
            factor = la.lu_factor(value)
            v = value @ basis
            back = la.lu_solve(factor, final.T, trans=1).T
            for j, c in enumerate(sim.c_ops):
                sparse = c.to('CSR').data.as_scipy()
                r = la.lu_solve(factor, sparse @ v)
                l = back @ (sparse @ value)
                right[index, j] = r.T.reshape(-1)
                left[index, j] = l.reshape(-1)
                one[index] += abs(np.trace(final @ r)) ** 2
        if channels:
            r = right.reshape(-1, d * sim.D)
            inner = np.zeros(len(times))
            for begin in range(0, len(times), 16):
                end = min(begin + 16, len(times))
                products = left[begin:end].reshape(-1, d * sim.D) @ r.T
                pairs = (abs(products) ** 2).reshape(end - begin, channels, len(times), channels).sum(axis=(1, 3))
                for local, i in enumerate(range(begin, end)):
                    if i:
                        inner[i] = simpson(pairs[local, :i + 1], x=times[:i + 1])
            first = simpson(one, x=times)
            second = simpson(inner, x=times)
        else:
            first = second = 0.0
        del left, right
        zero = abs(np.trace(final @ basis)) ** 2
        fidelity = float((d * (zero + first + second) / d ** 2 + effect['survival']) / (d + 1))
        difference = abs(fidelity - previous) if previous is not None else float('inf')
        print('JUMP CHANNEL', intervals, fidelity, 'quadrature change', difference, 'tail', effect['tail_bound'], flush=True)
        if difference < 5e-08:
            numerical = 10 * difference + effect['allowance']
            low = max(0.0, fidelity - numerical)
            high = min(1.0, fidelity + effect['tail_bound'] + numerical)
            return dict(fidelity=low, fidelity_interval=[low, high], leakage=max(0.0, 1 - effect['survival']), input_states=d * d, method='full_lindblad_two_jump_trace_and_counted_adjoint', value_kind='conservative_fidelity_lower_bound', quadrature_intervals=intervals, omitted_fidelity_bound=effect['tail_bound'], numerical_allowance=numerical, jump_count_diagnostics=effect)
        previous = fidelity
    raise ValueError('Two-jump quadrature has not converged')
