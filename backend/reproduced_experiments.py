import numpy as np
from .core import Segment
from .experiments.common import Result, new_fig
from .benchmark.catalog import register_experiment
from .acquisition_cache import reuse_acquisitions

def power_rabi_amplification(env, k, amp_factors, n_list, operation='x180', shots=50):
    factors = np.asarray(amp_factors, dtype=float)
    counts = np.asarray(n_list, dtype=int)
    base = env.cal.pi_amp[k] * (0.5 if operation == 'x90' else 1.0)
    if np.any(np.abs(base * factors) > 2):
        raise ValueError('Requested pulse amplitude exceeds platform range')
    circuits = [[Segment([env.x_pulse(k, amp=float(base * factor))]) for _ in range(int(count))] for count in counts for factor in factors]
    populations = [env.p1(iq[k], k) for iq in env.measure_many(circuits, shots, [k], name='power_rabi_amplification')]
    population = np.asarray(populations).reshape(len(counts), len(factors))
    fig, ax = new_fig(f'Power Rabi error amplification q{k}: {operation}', 'Amplitude factor', 'Number of pulses')
    mesh = ax.pcolormesh(factors, counts, population, shading='nearest', vmin=0, vmax=1)
    fig.colorbar(mesh, ax=ax, label='Measured P(1)')
    return Result('power_rabi_amplification', {'amp_factors': factors, 'n': counts, 'p1': population}, {'status': 'finished', 'params': {}, 'model': 'measured_2d_grid'}, {'shots': shots, 'operation': operation, 'base_amplitude': base, 'preparation': 'ground', 'axis_order': ['n', 'amp_factors']}, fig)

@reuse_acquisitions
def readout_frequency_scan(env, k, freqs, shots=1024):
    original = env.cal.readout_freq[k]
    means = [[], []]
    try:
        for freq in freqs:
            env.cal.readout_freq[k] = float(freq)
            circuits = [[], [Segment([env.x_pulse(k)])]]
            for state, circuit in enumerate(circuits):
                iq = env.measure(circuit, shots, [k], name='readout_frequency_scan')
                means[state].append(complex(np.mean(iq[k])))
    finally:
        env.cal.readout_freq[k] = original
    ground, excited = (np.asarray(means[0]), np.asarray(means[1]))
    distance = np.abs(ground - excited)
    fig, ax = new_fig(f'Readout frequency q{k}', 'Frequency (GHz)', 'Measured IQ separation')
    ax.plot(freqs, distance, 'o-')
    return Result('readout_frequency_scan', {'freq_GHz': np.asarray(freqs), 'distance': distance, 'iq_ground': ground, 'iq_excited': excited}, {'status': 'finished', 'params': {}, 'model': 'measured_iq_distance'}, {'shots': shots}, fig)

def install(n):
    register_experiment('power_rabi_amplification', power_rabi_amplification, {'description': 'Ground preparation, N repeated X180 or X90 pulses, measured P1 over pulse amplitude and count. No signed-error fit or automatic calibration update.', 'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['k', 'amp_factors', 'n_list'], 'properties': {'k': {'type': 'integer', 'minimum': 0, 'maximum': n - 1}, 'amp_factors': {'type': 'array', 'minItems': 5, 'maxItems': 257, 'items': {'type': 'number', 'minimum': 0.001, 'maximum': 2}}, 'n_list': {'type': 'array', 'minItems': 1, 'maxItems': 100, 'items': {'type': 'integer', 'minimum': 1, 'maximum': 200}}, 'operation': {'type': 'string', 'enum': ['x180', 'x90'], 'default': 'x180'}, 'shots': {'type': 'integer', 'minimum': 16, 'maximum': 4096, 'default': 50}}}}, lambda p: len(p['amp_factors']) * len(p['n_list']))
    register_experiment('readout_frequency_scan', readout_frequency_scan, {'description': 'Prepare ground and excited states at each readout frequency and return measured IQ distances; restore the original readout frequency after acquisition.', 'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['k', 'freqs'], 'properties': {'k': {'type': 'integer', 'minimum': 0, 'maximum': n - 1}, 'freqs': {'type': 'array', 'minItems': 5, 'maxItems': 257, 'items': {'type': 'number', 'minimum': 0.1, 'maximum': 15}}, 'shots': {'type': 'integer', 'minimum': 16, 'maximum': 4096, 'default': 1024}}}}, lambda p: 2 * len(p['freqs']))
