from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from .parametric import enabled
from .core import Segment
from .gates import compiled
from .experiments.common import Result, new_fig

def coupler_scan(env, edge, axis, values, preparation, basis, repetitions=1, shots=1024):
    if not enabled(env.dev):
        raise ValueError('Coupler scan requires the parametric apparatus')
    if tuple(edge) not in env.dev.edges:
        raise ValueError('An installed directed edge is required')
    bounds = {'amp': (0.0, 0.04), 'freq': (0.05, 1.0), 'duration': (10.0, 150.0)}
    lo, hi = bounds[axis]
    if any((not np.isfinite(v) or not lo <= v <= hi for v in values)):
        raise ValueError(f'{axis} values must lie in [{lo}, {hi}]')
    before = env.cal.copy()
    probabilities = []
    counts = []
    sequences = []
    try:
        for value in values:
            getattr(env.cal, 'coupler_' + axis)[tuple(edge)] = float(value)
            seq = []
            for q, state in enumerate(preparation):
                if state == '1':
                    seq.append(Segment([env.x_pulse(q)]))
                elif state == '+':
                    seq.append(Segment([env.y_pulse(q, -np.pi / 2)]))
                elif state == '+i':
                    seq.append(Segment([env.x_pulse(q, -np.pi / 2)]))
            frames = np.zeros(3)
            for _ in range(repetitions):
                block, advance = compiled(env, 'cz', tuple(edge), frames)
                seq.extend(block)
                frames += advance
            for q, axis_name in enumerate(basis):
                if axis_name == 'X':
                    seq.append(Segment([env.x_pulse(q, np.pi / 2, phase=np.pi / 2 + frames[q])]))
                elif axis_name == 'Y':
                    seq.append(Segment([env.x_pulse(q, np.pi / 2, phase=frames[q])]))
            sequences.append(seq)
    finally:
        env.cal = before
    for iq in env.measure_many(sequences, shots, list(range(3)), name='coupler_scan'):
        bits = np.array([env.assign(iq[q], q) for q in range(3)])
        count = np.bincount(np.ravel_multi_index(bits, (2,) * 3), minlength=8)
        counts.append(count)
        probabilities.append(count / shots)
    fig, ax = new_fig('Parametric CZ scan', axis, 'Measured joint probability')
    for k in range(8):
        ax.plot(values, np.asarray(probabilities)[:, k], '.-', label=format(k, '03b'))
    ax.legend(fontsize=7)
    return Result('coupler_scan', dict(values=values, counts=counts, probabilities=probabilities), dict(ok=True, params={}, model='measured_histogram_not_fidelity'), dict(shots=shots, axis=axis, includes_spam=True), fig)

def specification():

    def array(items, length):
        return dict(type='array', minItems=length, maxItems=length, items=items)
    props = dict(edge=array(dict(type='integer', minimum=0, maximum=2), 2), axis=dict(type='string', enum=['amp', 'freq', 'duration']), values=dict(type='array', minItems=1, maxItems=5, items=dict(type='number', minimum=0, maximum=150)), preparation=array(dict(type='string', enum=['0', '1', '+', '+i']), 3), basis=array(dict(type='string', enum=['X', 'Y', 'Z']), 3), repetitions=dict(type='integer', minimum=1, maximum=4, default=1), shots=dict(type='integer', minimum=16, maximum=4096, default=1024))
    return dict(description='Scan a short parametric CZ amplitude (GHz), frequency (GHz), or duration (ns). At most 5 points per call; split larger scans across requests. Finite preparation/analysis pulses, all spectators, leakage and sampled IQ are retained. Controls are restored after the scan.', parameters=dict(type='object', required=['edge', 'axis', 'values', 'preparation', 'basis'], additionalProperties=False, properties=props))

def configure(existing):
    from .benchmark.catalog import register_experiment
    register_experiment('coupler_scan', coupler_scan, specification(), lambda p: len(p['values']))
    out = deepcopy(existing)
    for name in ('cr_amp_scan', 'cr_hamiltonian_tomography', 'xy_plane_cnot_calibration', 'sizzle_zz_scan', 'calibrated_sizzle_ramsey', 'stark_shift_scan', 'ramsey_with_tones', 'spec_vs_power'):
        out.pop(name, None)
    out['coupler_scan'] = specification()
    for name, spec in out.items():
        props = spec['parameters']['properties']
        if name in ('qubit_spec', 'rabi_amp') and 'duration' in props:
            props['duration']['maximum'] = 150.0
        if name == 'qubit_spec':
            props['duration']['default'] = 100.0
        if name == 'rabi_time':
            props['durations']['items']['maximum'] = 150.0
        if 'n_list' in props:
            props['n_list']['maxItems'] = 9
            props['n_list']['items']['maximum'] = 16
        if 'reps' in props:
            props['reps']['maximum'] = 4
        for axis in ('freqs', 'amps', 'betas', 'angles'):
            if axis in props:
                props[axis]['maxItems'] = min(props[axis].get('maxItems', 31), 31)
        if 'delays' in props:
            props['delays']['maxItems'] = min(props['delays'].get('maxItems', 61), 61)
    if 'machine_probe' in out:
        out['machine_probe']['parameters']['properties']['repetitions']['maximum'] = 4
    return out
