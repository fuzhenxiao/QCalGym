from functools import lru_cache
import numpy as np
from scipy.optimize import curve_fit
from .core import Segment
from .experiments.common import Result, new_fig
from .benchmark.catalog import register_experiment

@lru_cache(maxsize=1)
def cliffords():
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    generators = [(identity - 1j * axis) / np.sqrt(2) for axis in (x, y)]
    group = [(identity, ())]
    cursor = 0
    while cursor < len(group):
        matrix, word = group[cursor]
        cursor += 1
        for axis, gate in enumerate(generators):
            candidate = gate @ matrix
            if not any((abs(np.trace(other.conj().T @ candidate)) > 2 - 1e-09 for other, _ in group)):
                group.append((candidate, word + (axis,)))
    assert len(group) == 24
    return tuple(group)

def single_qubit_rb(env, k, lengths, sequences=4, shots=256, seed=20261005):
    group = cliffords()
    rng = np.random.default_rng(seed)
    circuits, depth = ([], [])
    for length in lengths:
        for _ in range(sequences):
            matrix = np.eye(2, dtype=complex)
            word = ()
            for index in rng.integers(0, 24, size=length):
                gate, primitives = group[index]
                matrix = gate @ matrix
                word += primitives
            inverse = max(group, key=lambda item: abs(np.trace(item[0] @ matrix)))[1]
            circuits.append([Segment([env.x_pulse(k, np.pi / 2, phase=axis * np.pi / 2)]) for axis in word + inverse])
            depth.append(length)
    survival = [1 - env.p1(iq[k], k) for iq in env.measure_many(circuits, shots, [k], name='single_qubit_rb')]
    means = np.asarray(survival).reshape(len(lengths), sequences).mean(axis=1)
    fit = {'status': 'failed', 'params': {}, 'model': 'A*p**m+B'}
    if len(set(lengths)) >= 4:
        try:
            p, _ = curve_fit(lambda m, a, rate, b: a * rate ** m + b, lengths, means, p0=(0.5, 0.99, 0.5), bounds=([0, 0, 0], [1, 1, 1]), maxfev=10000)
            fit = {'status': 'finished', 'params': {'A': float(p[0]), 'p': float(p[1]), 'B': float(p[2]), 'error_per_clifford': float((1 - p[1]) / 2)}, 'model': 'A*p**m+B'}
        except (RuntimeError, ValueError):
            pass
    fig, ax = new_fig(f'Single-qubit RB q{k}', 'Clifford count', 'Ground-state survival')
    ax.plot(lengths, means, 'o-')
    return Result('single_qubit_rb', {'length': depth, 'survival': survival, 'mean_survival': means}, fit, {'shots': shots, 'sequences': sequences, 'seed': seed, 'includes_spam': True}, fig)

def install(n):
    register_experiment('single_qubit_rb', single_qubit_rb, {'description': 'Sample single-qubit Clifford sequences and their inverses; estimate RB decay from measured survival including SPAM.', 'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['k', 'lengths'], 'properties': {'k': {'type': 'integer', 'minimum': 0, 'maximum': n - 1}, 'lengths': {'type': 'array', 'minItems': 1, 'maxItems': 21, 'items': {'type': 'integer', 'minimum': 1, 'maximum': 64}}, 'sequences': {'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 4}, 'shots': {'type': 'integer', 'minimum': 16, 'maximum': 4096, 'default': 256}, 'seed': {'type': 'integer', 'minimum': 0, 'default': 20261005}}}}, lambda p: len(p['lengths']) * p['sequences'])
    register_experiment('drag_180_minus_180', drag_180_minus_180, {'description': 'Workflow-only: N repetitions of (x180, -x180) versus DRAG coefficient; measured P1 grid, no fit or update.', 'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['k', 'betas', 'n_list'], 'properties': {'k': {'type': 'integer', 'minimum': 0, 'maximum': n - 1}, 'betas': {'type': 'array', 'minItems': 3, 'maxItems': 101, 'items': {'type': 'number', 'minimum': -5, 'maximum': 5}}, 'n_list': {'type': 'array', 'minItems': 1, 'maxItems': 40, 'items': {'type': 'integer', 'minimum': 1, 'maximum': 100}}, 'shots': {'type': 'integer', 'minimum': 16, 'maximum': 4096, 'default': 100}}}}, lambda p: len(p['betas']) * len(p['n_list']), hidden=True)

def drag_180_minus_180(env, k, betas, n_list, shots=100):
    betas = np.asarray(betas, dtype=float)
    counts = np.asarray(n_list, dtype=int)
    circuits = []
    for count in counts:
        for beta in betas:
            pair = [Segment([env.x_pulse(k, drag=float(beta))]), Segment([env.x_pulse(k, phase=np.pi, drag=float(beta))])]
            circuits.append(pair * int(count))
    p1 = np.asarray([env.p1(iq[k], k) for iq in env.measure_many(circuits, shots, [k], name='drag_180_minus_180')])
    p1 = p1.reshape(len(counts), len(betas))
    fig, ax = new_fig(f'DRAG 180/-180 q{k}', 'DRAG coefficient', 'Measured P(1)')
    for row, count in zip(p1, counts):
        ax.plot(betas, row, 'o-', label=f'N={count}')
    ax.legend()
    return Result('drag_180_minus_180', {'betas': betas, 'n': counts, 'p1': p1}, {'status': 'finished', 'params': {}, 'model': 'measured_2d_grid'}, {'shots': shots, 'axis_order': ['n', 'betas'], 'includes_spam': True}, fig)
