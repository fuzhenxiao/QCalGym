from copy import deepcopy
from dataclasses import asdict
import json
import math
import numpy as np
RUNTIME_FIELDS = ('lo_freq', 'mixer_gain', 'mixer_skew', 'mixer_dc_i', 'mixer_dc_q', 'channel_delay', 'flux_bias', 'ampm_compensation', 'dressing_freq', 'dressing_amp')
EXTENSION_FIELDS = RUNTIME_FIELDS + ('dd_sequence', 'dd_pulses', 'dd_offset', 'dd_min_duration', 'readout_weights')
BOUNDS = {'lo_freq': (0.1, 15.0), 'mixer_gain': (0.2, 5.0), 'mixer_skew': (-1.4, 1.4), 'mixer_dc_i': (-2.0, 2.0), 'mixer_dc_q': (-2.0, 2.0), 'channel_delay': (-100.0, 100.0), 'flux_bias': (-0.5, 0.5), 'dd_pulses': (2, 32), 'dd_offset': (-0.45, 0.45), 'dd_min_duration': (0.0, 20000.0), 'ampm_compensation': (-20.0, 20.0), 'dressing_freq': (0.1, 15.0), 'dressing_amp': (0.0, 2.0)}
APPARATUS_CONTROLS = {'mixers': ('lo_freq', 'mixer_gain', 'mixer_skew', 'mixer_dc_i', 'mixer_dc_q'), 'delays': ('channel_delay',), 'ampm': ('ampm_compensation',), 'static_tls': ('dressing_freq', 'dressing_amp')}
IDEAL_APPARATUS = {'mixers': lambda: {'gain': 1.0, 'skew': 0.0, 'leakage': [0.0, 0.0]}, 'delays': lambda: 0.0, 'ampm': lambda: 0.0, 'static_tls': lambda: {'freq': 1.0, 'width': 0.001, 'g': 0.0}}

def apparatus_eligible(dev):
    return 'flux_coupler' not in dev.physics and (dev.n == 1 and 'cosine_transmon' in dev.physics or (dev.n <= 2 and all((q.n_levels == 3 for q in dev.qubits))))

def apparatus_controls_active(cal):
    return any((getattr(cal, name) for names in APPARATUS_CONTROLS.values() for name in names))

def with_apparatus(dev, cal):
    missing = [key for key, names in APPARATUS_CONTROLS.items() if key not in dev.physics and any((getattr(cal, name) for name in names)) and (not (key == 'static_tls' and 'cosine_transmon' in dev.physics))]
    if not missing:
        return dev
    if not apparatus_eligible(dev):
        raise ValueError('XY-source apparatus controls require a supported transmon source backend')
    out = deepcopy(dev)
    for key in missing:
        out.physics[key] = [IDEAL_APPARATUS[key]() for _ in range(dev.n)]
    return out

def dynamics_enabled(dev):
    return any((k in dev.physics for k in ('mixers', 'delays', 'flux', 'slow_noise', 'amplitude_compression', 'ampm', 'heating', 'parasitic_mode', 'cosine_transmon')))

def temporal_enabled(dev):
    return bool(dev.physics.get('readout', {}).get('temporal', False))

def require_legacy_segments(dev):
    if dynamics_enabled(dev):
        raise ValueError('Optional hardware requires Simulator.run on a complete sequence; segment/static-drive surrogates omit its physics')

def runtime_controls(cal):
    return {name: deepcopy(getattr(cal, name)) for name in RUNTIME_FIELDS}

def fingerprint(dev, cal):
    return json.dumps([dev.to_json(), runtime_controls(cal)], sort_keys=True)

def _number(x, name, low=None, high=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (not math.isfinite(x)):
        raise ValueError(f'{name} must be a finite number')
    if low is not None and x < low or (high is not None and x > high):
        raise ValueError(f'{name} outside [{low}, {high}]')

def validate_hardware(dev):
    p, n = (dev.physics, dev.n)
    if not isinstance(p, dict) or set(p) - {'mixers', 'delays', 'delay_kind', 'flux', 'slow_noise', 'readout', 'amplitude_compression', 'ampm', 'heating', 'parasitic_mode', 'static_tls', 'cosine_transmon'}:
        raise ValueError('Unknown optional hardware model')
    source_only = set(p) <= {'ampm', 'amplitude_compression'}
    allowed_levels = (2, 3) if source_only else (3,)
    if p and 'cosine_transmon' not in p and (n > 2 or any((q.n_levels not in allowed_levels for q in dev.qubits))):
        raise ValueError('Optional hardware requires one or two three-level transmons; source-only AM-AM/AM-PM also supports two-level qubits')
    for key in ('mixers', 'delays', 'flux', 'amplitude_compression', 'ampm'):
        if key in p and (not isinstance(p[key], list) or len(p[key]) != n):
            raise ValueError(f'{key} requires one entry per physical qubit/source')
    for beta in p.get('amplitude_compression', []):
        _number(beta, 'amplitude compression coefficient', 0.0, 100.0)
    for coefficient in p.get('ampm', []):
        _number(coefficient, 'AM-PM coefficient', -20.0, 20.0)
    for m in p.get('mixers', []):
        if set(m) - {'gain', 'skew', 'leakage'}:
            raise ValueError('Unknown mixer parameter')
        _number(m.get('gain', 1.0), 'mixer gain', 0.05, 10.0)
        _number(m.get('skew', 0.0), 'mixer skew', -1.4, 1.4)
        z = m.get('leakage', [0.0, 0.0])
        if not isinstance(z, list) or len(z) != 2:
            raise ValueError('LO leakage must be [real,imag] source amplitude')
        for x in z:
            _number(x, 'LO leakage', -2.0, 2.0)
    for d in p.get('delays', []):
        _number(d, 'physical delay', -100.0, 100.0)
    if p.get('delay_kind', 'envelope') not in ('envelope', 'rf'):
        raise ValueError('delay_kind must be envelope or rf')
    for f in p.get('flux', []):
        if f is None:
            continue
        allowed = {'ej_max', 'ec', 'asymmetry', 'offset', 'bias_min', 'bias_max', 'tls', 'flux_noise_std', 'chi_slope'}
        if set(f) - allowed or not {'ej_max', 'ec', 'asymmetry'} <= set(f):
            raise ValueError('Flux model requires ej_max, ec, asymmetry; unknown parameter rejected')
        for key in ('ej_max', 'ec'):
            _number(f[key], key, 1e-06)
        _number(f['asymmetry'], 'SQUID asymmetry', 0.01, 1.0)
        for key in ('offset', 'chi_slope'):
            _number(f.get(key, 0.0), key)
        _number(f.get('flux_noise_std', 0.0), 'flux noise std', 0.0, 0.01)
        low, high = (f.get('bias_min', -0.5), f.get('bias_max', 0.5))
        _number(low, 'bias_min', -0.5, 0.5)
        _number(high, 'bias_max', -0.5, 0.5)
        if low >= high:
            raise ValueError('Flux bias interval must have positive width')
        for tls in f.get('tls', []):
            if set(tls) != {'freq', 'width', 'rate'}:
                raise ValueError('TLS requires freq (GHz), width HWHM (GHz), peak rate (1/ns)')
            _number(tls['freq'], 'TLS frequency', 0.1, 15.0)
            _number(tls['width'], 'TLS width', 1e-09)
            _number(tls['rate'], 'TLS excess relaxation rate', 0.0)
    noise = p.get('slow_noise', {})
    if set(noise) - {'sigma', 'correlation'}:
        raise ValueError('Unknown slow_noise parameter')
    sig = noise.get('sigma', [0.0] * n)
    if len(sig) != n:
        raise ValueError('Noise sigma requires one GHz value per qubit')
    for x in sig:
        _number(x, 'noise sigma', 0.0, 0.001)
    corr = np.asarray(noise.get('correlation', np.eye(n)), dtype=float)
    if corr.shape != (n, n) or not np.isfinite(corr).all() or (not np.allclose(corr, corr.T)) or (not np.allclose(np.diag(corr), 1.0)) or (np.linalg.eigvalsh(corr).min() < -1e-12):
        raise ValueError('Noise correlation must be a positive semidefinite correlation matrix')
    from .nonlinear_readout import validate_readout
    validate_readout(dev)
    from .driven_baths import validate as validate_baths
    validate_baths(dev)
    from .cosine_transmon import validate as validate_cosine
    validate_cosine(dev)
    for entry in p.get('static_tls', []):
        if not isinstance(entry, dict) or set(entry) != {'freq', 'width', 'g'}:
            raise ValueError('Static TLS requires frequency, HWHM and coupling in GHz')
        for name, value in entry.items():
            _number(value, 'TLS ' + name, 0.0 if name == 'g' else 1e-12)
        if entry['g'] > entry['width'] / 5:
            raise ValueError('Eliminated TLS requires g <= width/5; coherent TLS needs an explicit auxiliary Hilbert space')
    if 'static_tls' in p and len(p['static_tls']) != n:
        raise ValueError('Static TLS requires one defect per qubit')

def validate_controls(dev, cal):
    validate_hardware(dev)
    from .instruments import validate as validate_instruments
    validate_instruments(dev, cal)
    from .nonlinear_readout import validate_amplitudes
    validate_amplitudes(dev, cal.readout_amplitude)
    if dev.n > 2 and any((getattr(cal, name) for name in EXTENSION_FIELDS)):
        raise ValueError('Optional controls currently support single/pair devices only')
    for name in EXTENSION_FIELDS:
        for k, v in getattr(cal, name).items():
            if type(k) is not int or not 0 <= k < dev.n:
                raise ValueError('Invalid control index')
            if name in BOUNDS:
                _number(v, name, *BOUNDS[name])
            if name == 'dd_sequence' and v not in ('off', 'cpmg', 'xy4', 'xy8'):
                raise ValueError('dd_sequence must be off, cpmg, xy4, or xy8')
            if name == 'dd_pulses' and (type(v) is not int or v % 2):
                raise ValueError('CPMG requires an even number of pulses')
            if name == 'flux_bias':
                f = dev.physics.get('flux', [None] * dev.n)[k]
                if f is None:
                    raise ValueError('This qubit has no flux actuator')
                _number(v, name, f.get('bias_min', -0.5), f.get('bias_max', 0.5))
            if any((name in names for names in APPARATUS_CONTROLS.values())) and v and (not apparatus_eligible(dev)):
                raise ValueError('XY-source apparatus controls require a supported transmon source backend')
            if name == 'readout_weights':
                if not temporal_enabled(dev):
                    raise ValueError('Weights require time-resolved readout')
                a = np.asarray(v, dtype=float)
                if a.ndim != 2 or a.shape[1] != 2 or (not 1 <= len(a) <= 256) or (not np.isfinite(a).all()) or (np.max(np.abs(a)) > 100) or (not np.any(a)):
                    raise ValueError('Weights require 1–256 finite, nonzero complex [real,imag] bins')
                from .temporal_readout import integration_weights
                integration_weights(v, dev.physics['readout'].get('bins', 64))

def flux_frequency(f, bias):
    x = np.pi * (bias + f.get('offset', 0.0))
    ej = f['ej_max'] * np.sqrt(np.cos(x) ** 2 + f['asymmetry'] ** 2 * np.sin(x) ** 2)
    return float(np.sqrt(8 * f['ec'] * ej) - f['ec'])

def effective_device(dev, cal):
    validate_controls(dev, cal)
    out = deepcopy(with_apparatus(dev, cal))
    if out.physics is not dev.physics:
        validate_hardware(out)
    for k, tls in enumerate(out.physics.get('static_tls', [])):
        q = out.qubits[k]
        amplitude = cal.dressing_amp.get(k, 0.0)
        detuning = q.freq - cal.dressing_freq.get(k, 5.2)
        omega = q.drive_scale * amplitude
        if amplitude:
            if min(abs(detuning), abs(detuning + q.anharm)) < max(0.03, 5 * abs(omega)):
                raise ValueError('Microwave dressed model requires dispersive drive: both detunings >= 5*Rabi rate and 30 MHz')
            q.freq += omega ** 2 * q.anharm / (2 * detuning * (detuning + q.anharm))
        delta = q.freq - tls['freq']
        denominator = delta ** 2 + tls['width'] ** 2
        rate = 4 * np.pi * tls['g'] ** 2 * tls['width'] / denominator
        q.freq += tls['g'] ** 2 * delta / denominator
        q.T1 = 1 / (1 / q.T1 + rate)
    for k, f in enumerate(dev.physics.get('flux', [])):
        if f is None:
            continue
        bias = cal.flux_bias.get(k, 0.0)
        _number(bias, 'flux_bias', f.get('bias_min', -0.5), f.get('bias_max', 0.5))
        q = out.qubits[k]
        q.freq, q.anharm = (flux_frequency(f, bias), -f['ec'])
        rate = 1.0 / dev.qubits[k].T1
        for tls in f.get('tls', []):
            rate += tls['rate'] / (1 + ((q.freq - tls['freq']) / tls['width']) ** 2)
        q.T1 = 1 / rate if rate else np.inf
        out.resonators[k].chi += f.get('chi_slope', 0.0) * (q.freq - flux_frequency(f, 0.0))
        if out.resonators[k].chi_levels is not None and f.get('chi_slope', 0.0):
            delta = out.resonators[k].chi - dev.resonators[k].chi
            out.resonators[k].chi_levels = [v + j * delta for j, v in enumerate(dev.resonators[k].chi_levels)]
    return out

def noise_factor(dev, controls):
    n = dev.n
    noise = dev.physics.get('slow_noise', {})
    sig = np.asarray(noise.get('sigma', [0.0] * n))
    covariance = np.outer(sig, sig) * np.asarray(noise.get('correlation', np.eye(n)))
    for k, f in enumerate(dev.physics.get('flux', [])):
        if f is not None and f.get('flux_noise_std', 0.0):
            b = controls.get('flux_bias', {}).get(k, 0.0)
            derivative = (flux_frequency(f, b + 1e-06) - flux_frequency(f, b - 1e-06)) / 2e-06
            covariance[k, k] += (derivative * f['flux_noise_std']) ** 2
    eig, vec = np.linalg.eigh(covariance)
    active = eig > 1e-24
    return vec[:, active] * np.sqrt(eig[active])

def public_apparatus(dev):
    if 'flux_coupler' in dev.physics:
        p = dev.physics['flux_coupler']
        laws = p['qubit_flux'] + [p['coupler_flux']]
        return {'units': 'GHz, ns, rad; flux in Phi0', 'n_qubits': 2, 'hardware': p['kind'], 'flux_ports': [k for k, f in enumerate(laws) if f], 'flux_ranges': {str(k): [f['bias_min'], f['bias_max']] for k, f in enumerate(laws) if f}, 'controls': ['qubit_bias', 'coupler_bias', 'mix', 'inverse_tau', 'inverse_tail'], 'readout_amplitude': {'default': 1.0, 'range': [0.0, 1000.0], 'power': 'proportional to amplitude squared'}, 'observations': 'Two data-qubit IQ channels; no direct coupler readout', 'initialization': 'Empty calibration; DC biases default to zero, mixing to identity, inverse filter disabled.'}
    return {'units': 'GHz, ns, rad; flux in Phi0', 'xy_source': 'Each drive source: IQ mixer with programmable LO, gain, skew and DC offsets; programmable channel delay; digital AM-PM predistortion; optional continuous dressing tone. Controls exist regardless of the hardware imperfections present.', 'lo_default_GHz': 5.0, 'flux_ranges': {str(k): [f.get('bias_min', -0.5), f.get('bias_max', 0.5)] for k, f in enumerate(dev.physics.get('flux', [])) if f is not None}, 'time_resolved_readout': temporal_enabled(dev), 'readout_amplitude': {'default': 1.0, 'range': [0.0, 1000.0], 'power': 'proportional to amplitude squared'}, 'dynamic_decoupling': dev.n <= 2, 'initialization': 'Absent flux_bias is zero; absent mixer gain/skew/DC and timing are 1/0/0/0.'}
