import numpy as np
from scipy.optimize import curve_fit
from scipy.special import ndtr
from .experiments.common import Result, new_fig
FIELDS = ('readout_time_of_flight', 'adc_offset_i', 'adc_offset_q', 'adc_gain')
BOUNDS = {'readout_time_of_flight': (0.0, 2000.0), 'adc_offset_i': (-2.0, 2.0), 'adc_offset_q': (-2.0, 2.0), 'adc_gain': (0.05, 20.0)}

def validate(dev, cal=None):
    entries = dev.physics.get('readout', {}).get('electronics', [{} for _ in dev.qubits])
    if not isinstance(entries, list) or len(entries) != dev.n:
        raise ValueError('Readout electronics requires one entry per receiver')
    bounds = {'delay_ns': (0.0, 2000.0), 'offset_i': (-1.0, 1.0), 'offset_q': (-1.0, 1.0), 'gain': (0.1, 10.0)}
    for p in entries:
        if not isinstance(p, dict) or set(p) - set(bounds):
            raise ValueError('Unknown readout electronics parameter')
        for name, value in p.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (not np.isfinite(value)) or (not bounds[name][0] <= value <= bounds[name][1]):
                raise ValueError('Invalid receiver ' + name)
    if cal is not None:
        for name in FIELDS:
            for k, value in getattr(cal, name).items():
                if type(k) is not int or not 0 <= k < dev.n or isinstance(value, bool) or (not isinstance(value, (int, float))) or (not np.isfinite(value)) or (not BOUNDS[name][0] <= value <= BOUNDS[name][1]):
                    raise ValueError('Invalid receiver control ' + name)

def receiver(dev, cal, k, duration):
    p = dev.physics.get('readout', {}).get('electronics', [{} for _ in dev.qubits])[k]

    def control(name, default):
        return getattr(cal, name, {}).get(k, default)
    delay = p.get('delay_ns', 0.0)
    start = control('readout_time_of_flight', 0.0)
    overlap = max(0.0, duration - abs(start - delay)) / duration
    gain = p.get('gain', 1.0) * control('adc_gain', 1.0)
    offset = gain * complex(p.get('offset_i', 0.0) + control('adc_offset_i', 0.0), p.get('offset_q', 0.0) + control('adc_offset_q', 0.0))
    return (overlap * gain, offset, gain)

def mixer_spectrum(env, source, rf_frequency=5.1, amplitude=0.2, shots=1024):
    from .extended_dynamics import iq_matrix, sidebands
    from .hardware import apparatus_eligible, IDEAL_APPARATUS
    model = env.dev.physics.get('mixers')
    if model is None:
        if not apparatus_eligible(env.dev):
            raise ValueError('No IQ mixer on this apparatus')
        model = [IDEAL_APPARATUS['mixers']() for _ in range(env.dev.n)]
    m, c = (model[source], env.cal)
    physical = iq_matrix(m.get('gain', 1.0), m.get('skew', 0.0))
    matrix = physical @ iq_matrix(c.mixer_gain.get(source, 1.0), c.mixer_skew.get(source, 0.0))
    main, image = sidebands(matrix)
    dc = physical @ np.array([c.mixer_dc_i.get(source, 0.0), c.mixer_dc_q.get(source, 0.0)])
    leakage = complex(*m.get('leakage', [0.0, 0.0])) + complex(*dc)
    lo = c.lo_freq.get(source, 5.0)
    freqs = np.array([lo, rf_frequency, 2 * lo - rf_frequency])
    if abs(rf_frequency - lo) < 0.001 or np.any((freqs < 0.1) | (freqs > 15.0)):
        raise ValueError('Instrument requires resolved in-range LO/RF/image bins')
    signal = np.array([leakage, amplitude * main, amplitude * image])
    voltage = signal[:, None] + 0.002 * (env.ro.rng.normal(size=(3, shots)) + 1j * env.ro.rng.normal(size=(3, shots)))
    powers = np.mean(abs(voltage) ** 2, axis=1)
    env.ledger.add('mixer_spectrum', 3 * shots, 1000.0, source=source)
    if hasattr(env, 'raw_arrays'):
        env.raw_arrays['instrument_voltage'] = voltage
    fig, ax = new_fig('Mixer output spectrum', 'frequency (GHz)', 'power (a.u.)')
    ax.bar(freqs, powers, width=0.005)
    return Result('mixer_spectrum', dict(frequency_GHz=freqs, power=powers), dict(status='not_fitted', params={}), dict(shots=shots, bins=['LO', 'RF', 'image']), fig)

def adc_trace(env, k, times, pulse_duration=400.0, shots=1024):
    t = np.asarray(times, float)
    if len(t) < 16 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('ADC sample times must be finite and strictly increasing')
    p = env.dev.physics.get('readout', {}).get('electronics', [{} for _ in env.dev.qubits])[k]
    _, offset, gain = receiver(env.dev, env.cal, k, pulse_duration)
    pulse = 0.2 + 0.1j
    edge = lambda delay: ndtr((t - delay) / 4.0) - ndtr((t - delay - pulse_duration) / 4.0)
    voltage = offset + gain * pulse * edge(p.get('delay_ns', 0.0))
    samples = voltage + 0.002 / np.sqrt(shots) * (env.ro.rng.normal(size=len(t)) + 1j * env.ro.rng.normal(size=len(t)))
    fit = dict(status='failed', params={}, model='known_loopback_envelope')
    try:

        def model(x, delay, g, oi, oq):
            z = complex(oi, oq) + g * pulse * edge(delay)
            return np.r_[z.real, z.imag]
        baseline = samples[:8].mean()
        projected = ((samples - baseline) / pulse).real
        height = max(float(projected.max()), 0.01)
        crossing = np.flatnonzero(projected > height / 2)
        seed_delay = float(np.clip(t[crossing[0]], 0.0, 2000.0)) if len(crossing) else 0.0
        values, _ = curve_fit(model, t, np.r_[samples.real, samples.imag], p0=[seed_delay, height, baseline.real, baseline.imag], bounds=([0.0, 0.001, -40.0, -40.0], [2000.0, 200.0, 40.0, 40.0]), maxfev=5000)
        residual = np.r_[samples.real, samples.imag] - model(t, *values)
        total = np.var(np.r_[samples.real, samples.imag])
        r2 = 1 - np.mean(residual ** 2) / max(total, 1e-30)
        fit = dict(status='finished' if r2 > 0.95 else 'failed', r2=float(r2), model='known_loopback_envelope', params=dict(arrival_ns=float(values[0]), gain=float(values[1]), offset_i=float(values[2]), offset_q=float(values[3])))
    except (ValueError, RuntimeError, FloatingPointError):
        pass
    env.ledger.add('adc_trace', shots, float(t[-1] - t[0]), qubit=k)
    if hasattr(env, 'raw_arrays'):
        env.raw_arrays['adc_mean_iq'] = samples
    fig, ax = new_fig(f'Receiver alignment q{k}', 'trigger time (ns)', 'ADC IQ')
    ax.plot(t, samples.real, label='I')
    ax.plot(t, samples.imag, label='Q')
    ax.legend()
    return Result('adc_trace', dict(time_ns=t, iq=samples), fit, dict(shots=shots), fig)

def install(dev):
    from .benchmark.catalog import register_experiment
    number = lambda lo, hi: dict(type='number', minimum=lo, maximum=hi)
    shots = dict(type='integer', minimum=16, maximum=20000, default=1024)
    specs = {}

    def add(name, fn, props, required, count):
        spec = dict(description=fn.__doc__, parameters=dict(type='object', properties=props, required=required, additionalProperties=False))
        register_experiment(name, fn, spec, count)
        specs[name] = spec
    add('adc_trace', adc_trace, dict(k=dict(type='integer', minimum=0, maximum=dev.n - 1), times=dict(type='array', minItems=16, maxItems=1024, items=number(-1000, 5000)), pulse_duration={**number(50, 2000), 'default': 400.0}, shots=shots), ['k', 'times'], lambda p: 1)
    from .hardware import apparatus_eligible
    if 'mixers' in dev.physics or apparatus_eligible(dev):
        add('mixer_spectrum', mixer_spectrum, dict(source=dict(type='integer', minimum=0, maximum=dev.n - 1), rf_frequency={**number(0.1, 15), 'default': 5.1}, amplitude={**number(0, 1), 'default': 0.2}, shots=shots), ['source'], lambda p: 3)
    return specs
