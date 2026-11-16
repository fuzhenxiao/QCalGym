from copy import deepcopy
from itertools import product
import numpy as np
from .core import Segment
from .experiments.common import Result, new_fig
from .hardware_probes import prepare_product
from .acquisition_cache import reuse_acquisitions

@reuse_acquisitions
def pump_probe(env, k, pump_freq, pump_amp, pump_duration, delays, probe='ground', shots=512):
    sequences = []
    for wait in delays:
        seq = [Segment([env.drive_pulse(k, pump_freq, pump_amp, pump_duration, shape='gaussian_square', rise=min(20.0, pump_duration / 4))]), float(wait)]
        if probe == 'excited':
            seq.append(Segment([env.x_pulse(k)]))
        if probe == 'coherence':
            seq.extend([Segment([env.x_pulse(k, np.pi / 2)]), 100.0, Segment([env.x_pulse(k, np.pi / 2)])])
        sequences.append(seq)
    values = []
    means = []
    for seq in sequences:
        iq = env.measure(seq, shots, [k], name='pump_probe')[k]
        values.append(env.p1(iq, k))
        means.append(iq.mean())
    fig, ax = new_fig('Pump / recovery / probe', 'Recovery wait (ns)', 'Assigned P(1)')
    ax.plot(delays, values, 'o-')
    return Result('pump_probe', dict(wait_ns=np.asarray(delays), p1=np.asarray(values), iq_mean=np.asarray(means)), {'status': 'not_fitted', 'params': {}}, {'includes_spam': True, 'probe': probe}, fig)

def phase_compensation_scan(env, k, coefficients, repetitions=3, shots=1024):
    before = env.cal.copy()
    values = []
    try:
        for coefficient in coefficients:
            env.cal.ampm_compensation[k] = float(coefficient)
            env.sync_physics()
            seq = []
            for _ in range(repetitions):
                seq.extend((Segment([env.x_pulse(k, angle)]) for angle in (np.pi / 2, np.pi, np.pi / 2)))
            iq = env.measure(seq, shots, [k], name='phase_compensation_scan')[k]
            values.append(env.p1(iq, k))
    finally:
        env.cal = before
        env.sync_physics()
    fig, ax = new_fig('Mixed-angle identity: X90 X180 X90', 'Phase correction (rad / amplitude²)', 'Assigned P(1)')
    ax.plot(coefficients, values, 'o-')
    return Result('phase_compensation_scan', dict(coefficient=np.asarray(coefficients), p1=np.asarray(values)), {'status': 'not_fitted', 'params': {}}, {'includes_spam': True, 'repetitions': repetitions, 'restored_calibration': True}, fig)

@reuse_acquisitions
def readout_joint_scan(env, k, amplitudes, frequencies, durations, shots=1024):
    before = env.cal.copy()
    rows = []
    preparations = list(product(('0', '1'), repeat=env.dev.n))
    try:
        for a, f, t in product(amplitudes, frequencies, durations):
            env.cal.readout_amplitude[k] = float(a)
            env.cal.readout_freq[k] = float(f)
            env.cal.readout_duration[k] = float(t)
            env.sync_physics()
            samples = {}
            for prep in preparations:
                samples[prep] = env.measure(prepare_product(env, list(prep)), shots, name='readout_joint_scan')[k]
            half = shots // 2
            centers = {prep: np.mean(x[:half]) for prep, x in samples.items()}
            direction = np.mean([x for p, x in centers.items() if p[k] == '1']) - np.mean([x for p, x in centers.items() if p[k] == '0'])
            angle = float(np.angle(direction))
            projected = {p: (x * np.exp(-1j * angle)).real for p, x in samples.items()}
            means = [np.mean(x[:half]) for x in projected.values()]
            thresholds = np.linspace(min(means), max(means), 101)
            contexts = list(product(('0', '1'), repeat=env.dev.n - 1))

            def errors(threshold, section):
                result = []
                for other in contexts:
                    p = list(other)
                    p.insert(k, '0')
                    q = p.copy()
                    q[k] = '1'
                    result.append((np.mean(projected[tuple(p)][section] > threshold) + np.mean(projected[tuple(q)][section] <= threshold)) / 2)
                return result
            threshold = float(min(thresholds, key=lambda threshold: max(errors(threshold, slice(None, half)))))
            rows.append([a, f, t, angle, threshold, max(errors(threshold, slice(half, None)))])
    finally:
        env.cal = before
        env.sync_physics()
    array = np.asarray(rows)
    columns = ('amplitude', 'frequency_GHz', 'duration_ns', 'angle', 'threshold', 'worst_context_error')
    points = dict(zip(columns, array.T))
    best = int(np.argmin(array[:, -1]))
    fig, ax = new_fig('Neighbour-conditioned readout', 'Scan point', 'Worst balanced assignment error')
    ax.plot(np.arange(len(rows)), array[:, -1], 'o-')
    return Result('readout_joint_scan', points, {'status': 'finished', 'params': dict(zip(columns, rows[best])), 'model': 'sampled_discriminator_grid', 'r2': None}, {'includes_spam': True, 'shots_per_preparation': shots, 'train_fraction': 0.5, 'restored_calibration': True}, fig)

def install(n):
    from .benchmark.catalog import register_experiment, catalog
    number = lambda lo, hi, default=None: dict(type='number', minimum=lo, maximum=hi, **{} if default is None else {'default': default})
    array = lambda lo, hi: dict(type='array', minItems=1, maxItems=101, items=number(lo, hi))
    k = dict(type='integer', minimum=0, maximum=n - 1)
    shots = dict(type='integer', minimum=32, maximum=4096, default=1024)
    specs = {'pump_probe': (pump_probe, dict(k=k, pump_freq=number(0.1, 15), pump_amp=number(-2, 2), pump_duration=number(0.1, 20000), delays=array(0, 200000), probe=dict(type='string', enum=['ground', 'excited', 'coherence'], default='ground'), shots={**shots, 'default': 512}), lambda p: len(p['delays'])), 'phase_compensation_scan': (phase_compensation_scan, dict(k=k, coefficients=array(-20, 20), repetitions=dict(type='integer', minimum=1, maximum=8, default=3), shots=shots), lambda p: len(p['coefficients'])), 'readout_joint_scan': (readout_joint_scan, dict(k=k, amplitudes=array(0.01, 1000), frequencies=array(0.1, 15), durations=array(1, 20000), shots=shots), lambda p: 2 ** n * len(p['amplitudes']) * len(p['frequencies']) * len(p['durations']))}
    for name, (function, props, counter) in specs.items():
        register_experiment(name, function, {'description': function.__doc__ or name.replace('_', ' '), 'parameters': dict(type='object', properties=props, required=[k for k, v in props.items() if 'default' not in v], additionalProperties=False)}, counter)
    return {name: catalog()[name] for name in specs}
