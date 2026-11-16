import numpy as np
import qutip as qt
import scipy.linalg as la
import scipy.sparse as sp
from scipy.integrate import simpson
from .core import Segment
from .jump_channel import setup, fundamental_samples

def counted_effects(sim, basis, prepared):
    w, rate, static, pieces, total = prepared
    D, d = (sim.D, basis.shape[1])
    nojump = -1j * (qt.spre(static) - qt.spost(static)) - 0.5 * (qt.spre(rate) + qt.spost(rate))
    jump = sum((qt.sprepost(c, c.dag()) for c in sim.c_ops), 0 * nojump)
    A = nojump.dag().to('CSR').data.as_scipy()
    J = jump.dag().to('CSR').data.as_scipy()
    generator = sp.block_diag([A, A, A, A + J], format='csr')
    ladder = sp.csr_matrix(([1.0, 1.0], ([1, 2], [0, 1])), shape=(4, 4))
    generator += sp.kron(ladder, J, format='csr')
    initial = np.zeros((4, D * D), complex)
    initial[0] = np.eye(D).reshape(-1, order='F')
    rotated = sim._R(total, w)[:, None] * basis
    initial[3] = (rotated @ rotated.conj().T).reshape(-1, order='F')
    state = initial.reshape(-1)

    def delta(op):
        adjoint = (-1j * (qt.spre(op) - qt.spost(op))).dag().to('CSR').data.as_scipy()
        return sp.kron(sp.eye(4), adjoint, format='csr')

    def constant(begin, end, operator):
        nonlocal state
        opts = {**sim.opts, 'method': 'dop853', 'normalize_output': False, 'store_states': False, 'store_final_state': True, 'nsteps': max(sim.opts['nsteps'], int((end - begin) * 1000))}
        state = qt.sesolve(qt.Qobj(1j * operator), qt.Qobj(state), [0.0, end - begin], options=opts).final_state.full().ravel()
    for begin, end, element, terms in reversed(pieces):
        if not isinstance(element, Segment):
            constant(begin, end, generator)
            continue
        h = [qt.Qobj(1j * generator)] + [[qt.Qobj(1j * delta(op)), lambda t, f=f: f(total - t)] for op, f in terms]
        feature = min((p.duration / p.n_sigma if p.shape == 'gaussian' else p.rise if p.shape == 'gaussian_square' else p.duration for p in element.pulses if p.duration > 0))
        opts = {**sim.opts, 'method': 'dop853', 'normalize_output': False, 'store_states': False, 'store_final_state': True, 'max_step': feature / 4, 'nsteps': max(sim.opts['nsteps'], int((end - begin) * 1000))}

        def integrate(a, b):
            nonlocal state
            state = qt.sesolve(h, qt.Qobj(state), [total - b, total - a], options=opts).final_state.full().ravel()
        common = all((abs(2 * np.pi * p.freq - w) < 1e-12 for p in element.pulses))
        if common and sim._is_flat_top(element):
            rise = element.pulses[0].rise
            integrate(end - rise, end)
            constant(begin + rise, end - rise, generator + delta(sim._constant_H(element, w) - static))
            integrate(begin, begin + rise)
        elif common and sim._is_constant(element):
            constant(begin, end, generator + delta(sim._constant_H(element, w) - static))
        else:
            integrate(begin, end)
    effects = state.reshape(4, D, D).transpose(0, 2, 1)
    missing = np.eye(d) - basis.conj().T @ sum(effects[:3]) @ basis
    missing = (missing + missing.conj().T) / 2
    allowance = max(1e-09, 100 * sim.opts['rtol'])
    values = np.maximum(0.0, la.eigvalsh(missing) + allowance)
    tail = float((np.sqrt(values).sum() / d) ** 2 * d / (d + 1))
    survival = float(np.trace(basis.conj().T @ effects[3] @ basis).real / d)
    if not np.isfinite([tail, survival]).all() or not -allowance <= survival <= 1 + allowance:
        raise FloatingPointError('Invalid counted-adjoint result')
    return dict(tail_bound=tail, missing_probability_bound=float(values.mean()), survival=survival, allowance=allowance, missing_eigenvalues=values.tolist())

def quadrature_times(sim, prepared, intervals):
    _, _, _, pieces, total = prepared
    boundaries = [0.0]
    for begin, end, element, terms in pieces:
        if isinstance(element, Segment) and sim._is_flat_top(element):
            rise = element.pulses[0].rise
            boundaries.extend([begin + rise, end - rise])
        boundaries.append(end)
    times = [0.0]
    for begin, end in zip(boundaries, boundaries[1:]):
        count = max(2, 2 * int(np.ceil((end - begin) / total * 8)))
        count *= intervals // 16
        times.extend(np.linspace(begin, end, count + 1)[1:])
    return np.asarray(times)

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
    collapse = [c.to('CSR').data.as_scipy() for c in sim.c_ops]
    for intervals in (16, 32, 64, 128, 256, 512, 1024, 2048):
        if intervals > max_intervals:
            break
        times = quadrature_times(sim, prepared, intervals)
        channels = len(sim.c_ops)
        left = np.empty((len(times), channels, d * sim.D), complex)
        right = np.empty_like(left)
        one = np.zeros(len(times))
        for index, value in enumerate(fundamental_samples(sim, sequence, times, prepared)):
            factor = la.lu_factor(value)
            v = value @ basis
            back = la.lu_solve(factor, final.T, trans=1).T
            for j, sparse in enumerate(collapse):
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
                products = left[begin:end].reshape(-1, d * sim.D) @ r[:end * channels].T
                pairs = (abs(products) ** 2).reshape(end - begin, channels, end, channels).sum(axis=(1, 3))
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
        if difference < 5e-08:
            numerical = 10 * difference + effect['allowance']
            low = max(0.0, fidelity - numerical)
            high = min(1.0, fidelity + effect['tail_bound'] + numerical)
            return dict(fidelity=low, fidelity_interval=[low, high], leakage=max(0.0, 1 - effect['survival']), input_states=d * d, method='full_lindblad_two_jump_aligned_quadrature_v1', quadrature_points=len(times), value_kind='conservative_fidelity_lower_bound', quadrature_intervals=intervals, omitted_fidelity_bound=effect['tail_bound'], numerical_allowance=numerical, jump_count_diagnostics=effect)
        previous = fidelity
    raise ValueError('Two-jump quadrature has not converged')
