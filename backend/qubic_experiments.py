import numpy as np
from .core import Segment
from .gates import compiled
from .experiments.common import Result, new_fig
from .benchmark.catalog import register_experiment

def cr_entanglement_scan(env, control, target, amps, repetitions=1, shots=1024):
    circuits, labels = ([], [])
    for amp in amps:
        for state in (0, 1):
            for axis in ('X', 'Y', 'Z'):
                seq = [Segment([env.x_pulse(control)])] if state else []
                seq += [env.cr_segment(control, target, amp=float(amp)) for _ in range(repetitions)]
                if axis != 'Z':
                    seq.append(Segment([env.x_pulse(target, np.pi / 2, phase=np.pi / 2 if axis == 'X' else 0.0)]))
                circuits.append(seq)
                labels.append((state, axis))
    values = {(state, axis): [] for state in (0, 1) for axis in ('X', 'Y', 'Z')}
    for label, iq in zip(labels, env.measure_many(circuits, shots, [target], name='cr_entanglement_scan')):
        values[label].append(1 - 2 * env.p1(iq[target], target))
    points = {'amp': list(amps), **{f'{axis}_ctrl{state}': val for (state, axis), val in values.items()}}
    fig, ax = new_fig('CR conditional Bloch vectors', 'CR amplitude', 'Measured expectation')
    for (state, axis), val in values.items():
        ax.plot(amps, val, label=f'{axis} | {state}')
    ax.legend()
    return Result('cr_entanglement_scan', points, {'status': 'finished', 'params': {}}, {'shots': shots, 'repetitions': repetitions, 'includes_spam': True}, fig)

def full_xy_scan(env, control, target, angles, shots=2048, prep_angles=(np.pi / 2, np.pi / 2)):
    circuits = []
    for angle in angles:
        seq = [Segment([env.x_pulse(control, float(prep_angles[0]))]), Segment([env.x_pulse(target, float(prep_angles[1]))])]
        block, frames = compiled(env, 'cnot', (control, target))
        seq += block
        seq += [Segment([env.x_pulse(q, np.pi / 2, phase=float(angle) + frames[q])]) for q in (control, target)]
        circuits.append(seq)
    population = []
    for iq in env.measure_many(circuits, shots, [control, target], name='full_xy_scan'):
        indices = 2 * env.assign(iq[control], control) + env.assign(iq[target], target)
        population.append((np.bincount(indices, minlength=4) / shots).tolist())
    fig, ax = new_fig('Full XY-plane CNOT measurement', 'Analysis angle (rad)', 'Joint probability')
    for j in range(4):
        ax.plot(angles, np.asarray(population)[:, j], label=format(j, '02b'))
    ax.legend()
    return Result('full_xy_scan', {'angle': list(angles), 'probabilities': population}, {'status': 'finished', 'params': {}}, {'shots': shots, 'includes_spam': True, 'gate': 'installed_cnot', **({} if tuple(prep_angles) == (np.pi / 2, np.pi / 2) else {'prep_angles': [float(a) for a in prep_angles]})}, fig)

def full_xy_scan_prepared(env, control, target, angles, prep_angles, shots=2048):
    return full_xy_scan(env, control, target, angles, shots=shots, prep_angles=tuple(prep_angles))

def install(n):
    if n < 2:
        return
    q = dict(type='integer', minimum=0, maximum=n - 1)
    shots = dict(type='integer', minimum=16, maximum=4096, default=1024)

    def add(name, fn, description, props, required, count, hidden=False):
        register_experiment(name, fn, dict(description=description, parameters=dict(type='object', additionalProperties=False, required=required, properties=dict(control=q, target=q, shots=shots, **props))), count, hidden=hidden)
    add('cr_entanglement_scan', cr_entanglement_scan, 'Measure XYZ target Bloch vectors after one or three raw CR pulses, control prepared in 0/1. No gate fidelity returned.', dict(amps=dict(type='array', minItems=5, maxItems=73, items=dict(type='number', minimum=0, maximum=2)), repetitions=dict(type='integer', enum=[1, 3], default=1)), ['control', 'target', 'amps'], lambda p: 6 * len(p['amps']))
    add('full_xy_scan', full_xy_scan, 'Prepare both qubits with X90, apply installed CNOT, sweep both XY analysis phases together; return four measured joint probabilities.', dict(angles=dict(type='array', minItems=9, maxItems=257, items=dict(type='number', minimum=-20, maximum=20))), ['control', 'target', 'angles'], lambda p: len(p['angles']))
    add('full_xy_scan_prepared', full_xy_scan_prepared, 'Workflow-only: as full_xy_scan, with control/target prepared by X(prep_angles) away from the poles.', dict(angles=dict(type='array', minItems=9, maxItems=257, items=dict(type='number', minimum=-20, maximum=20)), prep_angles=dict(type='array', minItems=2, maxItems=2, items=dict(type='number', minimum=-4, maximum=4))), ['control', 'target', 'angles', 'prep_angles'], lambda p: len(p['angles']), hidden=True)
