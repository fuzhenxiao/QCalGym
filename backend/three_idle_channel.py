import numpy as np
import qutip as qt
from .numerics import expm

def restrict_number_conserving(sim, basis):
    from copy import copy
    keep = np.flatnonzero(sim._Ntot() < 3.5)
    drop = np.setdiff1d(np.arange(sim.D), keep)
    if np.linalg.norm(basis[drop]) > 1e-10:
        raise ValueError('Computational basis is outside the invariant subspace')
    if any((np.linalg.norm(op.full()[np.ix_(drop, keep)]) > 1e-12 for op in [sim.H_static] + list(sim.c_ops))):
        raise ValueError('Generator leaves the invariant subspace')
    reduced = copy(sim)
    reduced.D = len(keep)
    reduced.dims = [len(keep)]
    reduced._ntot = sim._Ntot()[keep].copy()

    def project(op):
        return qt.Qobj(op.full()[np.ix_(keep, keep)])
    for name in ('a', 'ad', 'num', 'X', 'Y', 'c_ops'):
        setattr(reduced, name, [project(op) for op in getattr(sim, name)])
    for name in ('H_static', 'H_lin', 'H_anh', 'H_c'):
        setattr(reduced, name, project(getattr(sim, name)))
    return (reduced, basis[keep])

def score(sim, sequence, target, basis, frame):
    duration = float(sum(sequence))
    d = len(target)
    keep = np.flatnonzero(sim._Ntot() < 3.5)
    drop = np.setdiff1d(np.arange(sim.D), keep)
    if np.linalg.norm(basis[drop]) > 1e-10:
        raise ValueError('Computational basis is outside the idle invariant subspace')
    w = sim.w_frame[0]
    full_h = sim._static_frame_H(w).full()
    operators = [full_h] + [c.full() for c in sim.c_ops]
    if any((np.linalg.norm(op[np.ix_(drop, keep)]) > 1e-12 for op in operators)):
        raise ValueError('Idle generator does not preserve excitation subspace')
    h = qt.Qobj(full_h[np.ix_(keep, keep)])
    collapses = [qt.Qobj(c[np.ix_(keep, keep)]) for c in operators[1:]]
    b = basis[keep]
    encode = np.kron(b.conj(), b)
    propagator = expm(qt.liouvillian(h, collapses).full() * duration)
    rotation = sim._R(duration, w).conj()[keep]
    output_basis = rotation.conj()[:, None] * b * frame.conj()[None, :]
    decode = np.kron(output_basis.T, output_basis.conj().T)
    channel = decode @ propagator @ encode
    if not np.isfinite(channel).all():
        raise FloatingPointError('Nonfinite idle channel')
    diagonal = np.arange(d) * (d + 1)
    survival = float(channel[np.ix_(diagonal, diagonal)].sum().real)
    overlap = float(np.vdot(np.kron(target.conj(), target), channel).real)
    return dict(fidelity=(overlap + survival) / (d * (d + 1)), leakage=max(0.0, 1 - survival / d), input_states=d * d, reachable_hilbert_dimension=len(keep), channel_method='exact_full_lindblad_idle_invariant_excitation_subspace')
