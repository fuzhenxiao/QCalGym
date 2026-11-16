import numpy as np
from dataclasses import replace
from .core import Segment
from .gates import installed_idle
from .experiments.common import Result, new_fig
from .acquisition_cache import reuse_acquisitions

def prepare_product(env, preparation):
    if len(preparation) != env.dev.n:
        raise ValueError('One preparation per qubit is required')
    seq = []
    for k, state in enumerate(preparation):
        if state in ('1', '2'):
            seq.append(Segment([env.x_pulse(k)]))
        if state == '2':
            seq.append(Segment([env.x12_pulse(k)]))
        if state == '+':
            seq.append(Segment([env.y_pulse(k, -np.pi / 2)]))
        if state == '+i':
            seq.append(Segment([env.x_pulse(k, -np.pi / 2)]))
        if state not in ('0', '1', '2', '+', '+i'):
            raise ValueError('Unknown preparation')
    return seq

def analysis(env, seq, basis, frames=None):
    if len(basis) != env.dev.n:
        raise ValueError('One analysis basis per qubit is required')
    frames = np.zeros(env.dev.n) if frames is None else frames
    for k, axis in enumerate(basis):
        if axis == 'X':
            seq.append(Segment([env.x_pulse(k, np.pi / 2, phase=np.pi / 2 + frames[k])]))
        elif axis == 'Y':
            seq.append(Segment([env.x_pulse(k, -np.pi / 2, phase=frames[k])]))
        elif axis != 'Z':
            raise ValueError('Unknown analysis basis')

def state_readout(env, preparation, shots=1024, time_resolved=False):
    seq = prepare_product(env, preparation)
    if time_resolved:
        iq = env.measure_traces(seq, shots, name='state_readout_trace')
        fig, ax = new_fig('Readout time records', 'time (ns)', 'Mean IQ')
        points = {}
        for k, v in iq.items():
            times = (np.arange(v.shape[1]) + 0.5) * env.cal.readout_duration[k] / v.shape[1]
            points[f'q{k}_time_ns'] = times
            points[f'q{k}_mean_iq'] = v.mean(axis=0)
            ax.plot(times, v.mean(axis=0).real, label=f'q{k} I')
            ax.plot(times, v.mean(axis=0).imag, '--', label=f'q{k} Q')
    else:
        iq = env.measure(seq, shots, name='state_readout')
        fig, ax = new_fig('Conditioned product-state readout', 'I', 'Q')
        points = {f'q{k}_mean_iq': v.mean() for k, v in iq.items()}
        for k, v in iq.items():
            ax.scatter(v.real, v.imag, s=3, alpha=0.2, label=f'q{k}')
    ax.legend()
    return Result('state_readout', points, {'status': 'not_fitted', 'params': {}}, {'preparation': preparation, 'includes_spam': True, 'time_resolved': time_resolved, 'shots': shots}, fig)

@reuse_acquisitions
def idle_response(env, preparation, basis, delays, installed=False, shots=1024):
    values, iq_mean = ([], [])
    for delay in delays:
        seq = prepare_product(env, preparation)
        block, frames = installed_idle(env, float(delay)) if installed else ([float(delay)], np.zeros(env.dev.n))
        seq.extend(block)
        analysis(env, seq, basis, frames)
        iq = env.measure(seq, shots, name='idle_response')
        values.append([env.p1(iq[k], k) for k in range(env.dev.n)])
        iq_mean.append([iq[k].mean() for k in range(env.dev.n)])
    fig, ax = new_fig('Idle response', 'delay (ns)', 'Assigned P(1)')
    for k in range(env.dev.n):
        ax.plot(delays, np.asarray(values)[:, k], 'o-', label=f'q{k}')
    ax.legend()
    return Result('idle_response', {'delay_ns': np.asarray(delays), 'p1': np.asarray(values), 'iq_mean': np.asarray(iq_mean)}, {'status': 'not_fitted', 'params': {}}, {'includes_spam': True, 'installed_idle': installed}, fig)

def drive_response(env, source, freqs, amp, duration, preparation, basis, phase=0.0, shots=1024):
    if not 0 <= source < env.dev.n:
        raise ValueError('Source not present on this device')
    values = []
    for f in freqs:
        seq = prepare_product(env, preparation)
        seq.append(Segment([env.drive_pulse(source, float(f), amp, duration, phase=phase, shape='gaussian_square', rise=min(20.0, duration / 4))]))
        analysis(env, seq, basis)
        iq = env.measure(seq, shots, name='drive_response')
        values.append([env.p1(iq[k], k) for k in range(env.dev.n)])
    fig, ax = new_fig('All-qubit response to one commanded source', 'RF frequency (GHz)', 'Assigned P(1)')
    for k in range(env.dev.n):
        ax.plot(freqs, np.asarray(values)[:, k], 'o-', label=f'q{k}')
    ax.legend()
    return Result('drive_response', {'freq_GHz': np.asarray(freqs), 'p1': np.asarray(values)}, {'status': 'not_fitted', 'params': {}}, {'source': source, 'includes_spam': True, 'lo_GHz': env.cal.lo_freq.get(env.cal.drive_line.get(source, source), 5.0)}, fig)

def bias_scan(env, k, biases, values, observable='spectroscopy', drive_freqs=None, amp=0.02, duration=1000.0, shots=512):
    from .hardware import validate_controls
    before = env.cal.copy()
    bounds = (0.1, 15.0) if observable == 'spectroscopy' else (0.0, 200000.0)
    if any((not bounds[0] <= v <= bounds[1] for v in values)):
        raise ValueError('Bias scan axis outside the selected observable range')
    if observable == 'relaxation' and (drive_freqs is None or len(drive_freqs) != len(biases)):
        raise ValueError('Relaxation scan requires one measured drive_freq per bias')
    grid = []
    try:
        for b in biases:
            candidate = before.copy()
            candidate.flux_bias[k] = b
            validate_controls(env.dev, candidate)
        for row, b in enumerate(biases):
            env.cal = before.copy()
            env.cal.flux_bias[k] = b
            if observable == 'relaxation':
                env.cal.drive_freq[k] = drive_freqs[row]
            env.sync_physics()
            line = []
            for value in values:
                if observable == 'spectroscopy':
                    seq = [Segment([env.drive_pulse(k, value, amp, duration, shape='gaussian_square', rise=duration / 5)])]
                elif observable == 'relaxation':
                    seq = [Segment([env.x_pulse(k)]), float(value)]
                else:
                    raise ValueError('Unknown bias scan observable')
                iq = env.measure(seq, shots, [k], name='bias_scan')[k]
                line.append(iq.mean())
            grid.append(line)
    finally:
        env.cal = before
        env.sync_physics()
    fig, ax = new_fig('Bias-dependent measured IQ', 'RF frequency (GHz)' if observable == 'spectroscopy' else 'wait (ns)', 'DC flux (Phi0)')
    array = np.asarray(grid)
    ax.pcolormesh(values, biases, np.abs(array - array[:, :1]), shading='auto')
    return Result('bias_scan', {'bias': np.asarray(biases), 'axis': np.asarray(values), 'iq_mean': array}, {'status': 'not_fitted', 'params': {}}, {'observable': observable, 'restored_calibration': True, 'includes_spam': True}, fig)

def spectator_echo_timing(env, probe, spectator, offsets, duration=400.0, shots=4096):
    if env.dev.n != 2 or {probe, spectator} != {0, 1}:
        raise ValueError('Distinct probe and spectator on a pair are required')
    offsets = np.asarray(offsets, float)
    width = env.cal.pi_duration[spectator]
    starts = (duration - width) / 2 + offsets
    if np.any(starts < 0) or np.any(starts + width > duration):
        raise ValueError('Spectator pulse must fit inside the nominal Ramsey interval')
    xy = np.empty((2, len(offsets), 2))
    for b in (0, 1):
        for j, start in enumerate(starts):
            for a, axis in enumerate(('X', 'Y')):
                seq = [Segment([env.x_pulse(spectator)]) if b else float(width), 200.0, Segment([env.y_pulse(probe, -np.pi / 2)]), Segment([replace(env.x_pulse(spectator), start=float(start))]), float(duration - start - width)]
                basis = ['Z', 'Z']
                basis[probe] = axis
                analysis(env, seq, basis)
                iq = env.measure(seq, shots, [probe], name='spectator_echo_timing')
                xy[b, j, a] = 1 - 2 * np.mean(env.assign(iq[probe], probe))
    phases = np.arctan2(xy[:, :, 1], xy[:, :, 0])
    difference = np.unwrap(np.angle(np.exp(1j * (phases[1] - phases[0]))))
    slope, intercept = np.polyfit(offsets, difference, 1)
    residual = difference - (slope * offsets + intercept)
    root = float(-intercept / slope) if abs(slope) > 1e-08 else None
    fit = dict(status='finished' if root is not None else 'unresolved', model='differential_phase_line', params=dict(slope_rad_per_ns=float(slope), intercept_rad=float(intercept), echo_center_ns=root, residual_rms_rad=float(np.sqrt(np.mean(residual ** 2)))), errors={})
    fig, ax = new_fig('Spectator echo timing: measured differential phase', 'Commanded echo offset (ns)', 'Phase difference (rad)')
    ax.plot(offsets, difference, 'o', label='Measured spectator 1 − 0')
    ax.plot(offsets, slope * offsets + intercept, '-', label='Linear fit')
    ax.axhline(0, color='gray', ls='--')
    ax.legend()
    ax.grid(alpha=0.25)
    return Result('spectator_echo_timing', dict(offset_ns=offsets, XY=xy, phase_rad=phases, differential_phase_rad=difference), fit, dict(shots=shots, includes_spam=True, finite_echo=True), fig)

def install(dev):
    from .benchmark.catalog import register_experiment
    number = lambda lo, hi: {'type': 'number', 'minimum': lo, 'maximum': hi}
    arr = lambda item, lo=1, hi=129: {'type': 'array', 'items': item, 'minItems': lo, 'maxItems': hi}
    states = arr({'type': 'string', 'enum': ['0', '1', '2', '+', '+i']}, 1, 2)
    basis = arr({'type': 'string', 'enum': ['X', 'Y', 'Z']}, 1, 2)
    shots = {'type': 'integer', 'minimum': 16, 'maximum': 4096, 'default': 1024}
    specs = {}

    def add(name, function, description, properties, required, count):
        spec = {'description': description, 'parameters': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}}
        register_experiment(name, function, spec, count)
        specs[name] = spec
    add('state_readout', state_readout, 'Prepare a product state and acquire raw IQ, optionally time-bin IQ; includes preparation errors.', {'preparation': states, 'shots': shots, 'time_resolved': {'type': 'boolean', 'default': False}}, ['preparation'], lambda p: 1)
    add('idle_response', idle_response, 'Measure product-Pauli responses versus bare or installed idle; installed mode includes finite DD and hold settings.', {'preparation': states, 'basis': basis, 'delays': arr(number(0, 200000)), 'shots': shots, 'installed': {'type': 'boolean', 'default': False}}, ['preparation', 'basis', 'delays'], lambda p: len(p['delays']))
    add('drive_response', drive_response, 'Observe both qubits while scanning one RF source. Change lo_freq separately to move the image at fixed RF.', {'source': {'type': 'integer', 'minimum': 0, 'maximum': dev.n - 1}, 'freqs': arr(number(0.1, 15)), 'amp': number(-2, 2), 'duration': number(1, 20000), 'phase': {**number(-20, 20), 'default': 0.0}, 'preparation': states, 'basis': basis, 'shots': shots}, ['source', 'freqs', 'amp', 'duration', 'preparation', 'basis'], lambda p: len(p['freqs']))
    if dev.n == 2 and 'flux_coupler' not in dev.physics:
        add('spectator_echo_timing', spectator_echo_timing, 'Ramsey on one qubit with a finite spectator X180 at scanned offsets; both spectator inputs and X/Y quadratures. Differential phase distinguishes common detuning from relative timing.', {'probe': {'type': 'integer', 'minimum': 0, 'maximum': 1}, 'spectator': {'type': 'integer', 'minimum': 0, 'maximum': 1}, 'offsets': arr(number(-1000, 1000), 3, 41), 'duration': {**number(100, 10000), 'default': 400.0}, 'shots': shots}, ['probe', 'spectator', 'offsets'], lambda p: 4 * len(p['offsets']))
    if any(dev.physics.get('flux', [])):
        add('bias_scan', bias_scan, 'Acquire a bias/frequency or bias/wait grid. Relaxation requires user-supplied measured carriers. Restores all controls after the scan.', {'k': {'type': 'integer', 'minimum': 0, 'maximum': dev.n - 1}, 'biases': arr(number(-0.5, 0.5), 2, 21), 'values': arr(number(0, 200000), 2, 129), 'observable': {'type': 'string', 'enum': ['spectroscopy', 'relaxation'], 'default': 'spectroscopy'}, 'drive_freqs': {**arr(number(0.1, 15), 2, 21), 'type': ['array', 'null'], 'default': None}, 'amp': {**number(0, 2), 'default': 0.02}, 'duration': {**number(1, 20000), 'default': 1000.0}, 'shots': {**shots, 'default': 512}}, ['k', 'biases', 'values'], lambda p: len(p['biases']) * len(p['values']))
    return specs
