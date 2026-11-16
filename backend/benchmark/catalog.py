import inspect
import math
from copy import deepcopy
import numpy as np
from .. import experiments
DESCRIPTIONS = {'resonator_spec': 'Scan readout frequency; return averaged S21 magnitude and Lorentzian fit.', 'qubit_spec': 'Scan drive frequency; return IQ response, P1 and Lorentzian fit.', 'rabi_time': 'Scan pulse duration and fit damped oscillations.', 'rabi_amp': 'Scan Gaussian pulse amplitude and fit oscillations.', 'gmm_readout': 'Prepare 0/1 and fit IQ clouds; optional explicit discriminator update.', 'ramsey': 'Scan Ramsey delay and fit damped oscillations; the fitted frequency is unsigned.', 't1': 'Scan relaxation delay and fit exponential T1.', 'echo': 'Scan Hahn echo delay and fit exponential coherence decay.', 'pingpong': 'X90 followed by N X180 gates, measured versus N.', 'drag_calibration': 'Scan DRAG coefficient with two phase-sensitive sequences.', 'repeated_gate': 'N repeated X90 gates, measured versus N.', 'conditional_ramsey': 'Ramsey traces with the spectator prepared in 0 and in 1; fitted frequencies are unsigned.', 'cr_amp_scan': 'Scan CR amplitude for control in 0/1 and fit target oscillations.', 'cr_hamiltonian_tomography': 'Measure standard Pauli target XYZ versus CR duration for both control states; fit Bloch vectors w0,w1 in rad/ns. I*=(w1+w0)/(4pi), Z*=(w1-w0)/(4pi) in GHz; Z* uses control1-minus-control0, opposite the standard control-Z convention. For undriven diagonal terms, conditional transition shift = -2*ZZ_GHz.', 'gmm3_readout': 'Prepare 0/1/2 and fit IQ clusters; optional explicit calibration update.', 'leakage_vs_pulses': 'Level-2 population versus pulse count, assigned with the installed three-state readout calibration.', 'spec_vs_power': 'Repeat spectroscopy over drive amplitudes and return per-power traces.', 'xy_plane_cnot_calibration': 'Scan standard target XY analysis axes after CR; fit a planar rotation axis modulo pi assuming approximately opposite conditional rotations.', 'ramsey_with_tones': 'Run Ramsey with specified (line, frequency GHz, amplitude, phase rad) tones during delays.', 'stark_shift_scan': 'Measure differential Ramsey frequency shift relative to the first supplied amplitude.', 'sizzle_zz_scan': 'Measure conditional Ramsey ZZ versus simultaneous tone amplitude.', 'calibrated_sizzle_ramsey': 'Run tone Ramsey using the current stored siZZle settings.'}
ARRAYS = {'freqs', 'amps', 'durations', 'delays', 'betas', 'n_list', 'angles'}
INTEGERS = {'k', 'i', 'j', 'control', 'target', 'shots', 'n_avg', 'reps'}
QUBITS = {'k', 'i', 'j', 'control', 'target'}
ARRAY_BOUNDS = {'freqs': (0.1, 15), 'amps': (-2, 2), 'durations': (0, 200000), 'delays': (0, 200000), 'betas': (-5, 5), 'n_list': (0, 256), 'angles': (-20, 20)}
_REGISTERED = {}
_HIDDEN = set()
CHAIN_SCAN_LIMITS = {'sizzle_zz_scan': 20, 'stark_shift_scan': 10}

def with_device_limits(experiment_catalog, n_qubits, topology):
    out = deepcopy(experiment_catalog)
    if n_qubits != 3 or topology != 'chain3':
        return out
    for name, limit in CHAIN_SCAN_LIMITS.items():
        if name not in out:
            continue
        spec = out[name]
        formula = '2 * len(amps) * len(delays)' if name == 'sizzle_zz_scan' else 'len(amps) * len(delays)'
        note = f' Three-qubit chain limit: {formula} <= {limit} acquisitions per request, excluding shots. Split larger scans across requests.'
        if note not in spec['description']:
            spec['description'] += note
        spec['max_acquisitions'] = limit
    return out

def register_experiment(name, function, specification, acquisition_counter, hidden=False):
    if name in experiments.__all__:
        raise ValueError('Cannot replace a built-in experiment')
    _REGISTERED[name] = (deepcopy(specification), acquisition_counter)
    if hidden:
        _HIDDEN.add(name)
    setattr(experiments, name, function)

def parameter_schema(name, param, experiment):
    default = param.default
    if name in ARRAYS:
        lo, hi = ARRAY_BOUNDS[name]
        fit_minimum = 5 if name != 'amps' or experiment in {'rabi_amp', 'cr_amp_scan'} else 2
        schema = {'type': 'array', 'minItems': fit_minimum, 'maxItems': 257, 'items': {'type': 'integer' if name == 'n_list' else 'number', 'minimum': lo, 'maximum': hi}}
    elif name == 'tones':
        schema = {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'number'}}}
    elif name == 'spectator' and experiment != 'conditional_ramsey':
        schema = {'type': 'object', 'additionalProperties': {'type': 'integer', 'enum': [0, 1]}, 'description': 'Map qubit index strings to preparation level 0 or 1.'}
    elif name in INTEGERS or name == 'spectator':
        schema = {'type': 'integer', 'minimum': 0, 'maximum': 3}
        if name in {'shots', 'n_avg'}:
            schema.update(minimum=16, maximum=20000)
        if name == 'reps':
            schema.update(minimum=1, maximum=128)
    elif name in {'excited', 'update_cal'}:
        schema = {'type': 'boolean'}
    elif name in {'shape', 'gate', 'name'}:
        schema = {'type': 'string', 'maxLength': 80}
        if name == 'shape':
            schema['enum'] = ['gaussian', 'square', 'gaussian_square']
        if name == 'gate':
            schema['enum'] = ['x90']
    else:
        schema = {'type': 'number', 'minimum': -20, 'maximum': 20}
        if 'duration' in name or name == 'rise':
            schema.update(minimum=0.01, maximum=200000)
        elif 'freq' in name:
            schema.update(minimum=0.1, maximum=15)
        elif 'amp' in name:
            schema.update(minimum=-2, maximum=2)
        elif name == 'virtual_detuning':
            schema.update(minimum=-0.1, maximum=0.1)
        elif name == 'drag':
            schema.update(minimum=-5, maximum=5)
    if default is not inspect.Parameter.empty:
        schema['default'] = default
        if default is None:
            schema['type'] = [schema['type'], 'null']
    return schema

def catalog(n_qubits=None, topology=None, include_hidden=False):
    out = {}
    for name in experiments.__all__:
        fn = getattr(experiments, name)
        params = dict(inspect.signature(fn).parameters)
        if name == 'calibrated_sizzle_ramsey':
            params.pop('kw')
            params.update({k: v for k, v in inspect.signature(experiments.ramsey_with_tones).parameters.items() if k in {'virtual_detuning', 'shots', 'spectator', 'name', 'rise'}})
            params['rise'] = params['rise'].replace(default=None)
        properties = {k: parameter_schema(k, p, name) for k, p in params.items() if k not in {'env', 'include_reference'}}
        if n_qubits is not None:
            for key in QUBITS | {'spectator'}:
                if key in properties and properties[key]['type'] == 'integer':
                    properties[key]['maximum'] = n_qubits - 1
        if 'update_cal' in properties:
            properties['update_cal']['default'] = False
        out[name] = {'description': DESCRIPTIONS[name], 'parameters': {'type': 'object', 'properties': properties, 'required': [k for k, v in properties.items() if 'default' not in v], 'additionalProperties': False}}
    out.update({name: deepcopy(value[0]) for name, value in _REGISTERED.items() if include_hidden or name not in _HIDDEN})
    out.pop('coupler_scan', None)
    return with_device_limits(out, n_qubits, topology)

def validate(value, schema, path='params'):
    types = schema.get('type')
    types = types if isinstance(types, list) else [types]
    checks = {'null': value is None, 'boolean': isinstance(value, bool), 'integer': isinstance(value, int) and (not isinstance(value, bool)), 'number': isinstance(value, (int, float)) and (not isinstance(value, bool)), 'string': isinstance(value, str), 'array': isinstance(value, list), 'object': isinstance(value, dict)}
    if not any((checks.get(t, False) for t in types)):
        raise ValueError(f'{path}: expected {types}')
    if value is None:
        return
    if isinstance(value, (float, int)) and (not isinstance(value, bool)):
        if not math.isfinite(value) or value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            raise ValueError(f'{path}: outside finite allowed range')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f"{path}: expected one of {schema['enum']}")
    if isinstance(value, list):
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', 100000):
            raise ValueError(f'{path}: invalid array length')
        for i, item in enumerate(value):
            validate(item, schema['items'], f'{path}[{i}]')
    if isinstance(value, str) and len(value) > schema.get('maxLength', 100000):
        raise ValueError(f'{path}: string too long')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                raise ValueError(f'{path}: missing {key}')
        for key, item in value.items():
            subschema = props.get(key, schema.get('additionalProperties', False))
            if not isinstance(key, str) or subschema is False:
                raise ValueError(f'{path}: unknown parameter {key}')
            if isinstance(subschema, dict):
                validate(item, subschema, f'{path}.{key}')

def prepare(name, params, env):
    specs = catalog(n_qubits=env.dev.n, topology=env.dev.topology, include_hidden=True)
    from ..parametric import enabled
    if enabled(env.dev):
        from ..parametric_probe import configure
        specs = configure(specs)
    if name not in specs:
        raise ValueError(f'Unknown experiment: {name}')
    schema = specs[name]['parameters']
    validate(params, schema)
    out = {k: deepcopy(s['default']) for k, s in schema['properties'].items() if 'default' in s}
    out.update(deepcopy(params))
    for k, value in out.items():
        if k in QUBITS or (k == 'spectator' and isinstance(value, int)):
            if value >= env.dev.n:
                raise ValueError(f'{k}: qubit not on this device')
        if k in ARRAYS:
            if k not in {'amps'} or name == 'rabi_amp':
                if any((b <= a for a, b in zip(value, value[1:]))):
                    raise ValueError(f'{k}: scan points must be strictly increasing')
            if k in {'freqs', 'durations', 'delays', 'n_list', 'angles', 'betas'} and len(value) < 5 and (name not in _REGISTERED):
                raise ValueError(f'{k}: at least five points needed for standard fits')
            if k == 'amps' and name in {'rabi_amp', 'cr_amp_scan'} and (len(value) < 5):
                raise ValueError('amps: at least five points needed for cosine fits')
    spec = out.get('spectator')
    if isinstance(spec, dict):
        try:
            spec = {int(k): v for k, v in spec.items()}
        except ValueError:
            raise ValueError('spectator keys must be qubit indices') from None
        if any((k < 0 or k >= env.dev.n or k == out.get('k', out.get('i')) for k in spec)):
            raise ValueError('spectator must be a different device qubit')
        out['spectator'] = spec
    for a, b in (('control', 'target'), ('i', 'j'), ('k', 'spectator')):
        if a in out and isinstance(out.get(b), int) and (out[a] == out[b]):
            raise ValueError(f'{a} and {b} must differ')
    for tone in out.get('tones', []):
        line, freq, amp, phase = tone
        if int(line) != line or not 0 <= line < env.dev.n or (not 0.1 <= freq <= 15) or (abs(amp) > 2) or (abs(phase) > 20):
            raise ValueError('Invalid tone tuple')
        tone[0] = int(line)
    if name == 'rabi_time' and any((0 < d < 2 * out['rise'] for d in out['durations'])):
        raise ValueError('Nonzero Rabi duration must be at least twice rise')
    if name in {'cr_amp_scan', 'cr_hamiltonian_tomography', 'xy_plane_cnot_calibration'}:
        edge = (out['control'], out['target'])
        if edge not in env.cal.cr_amp:
            raise ValueError('CR requires a calibrated device edge')
        rise = out.get('rise') or env.cal.cr_rise.get(edge, 20.0)
        durations = out.get('durations', [out.get('duration') or out.get('cr_duration') or env.cal.cr_duration[edge]])
        if any((d < 2 * rise for d in durations)):
            raise ValueError('CR durations must be at least twice rise')
    if name == 'calibrated_sizzle_ramsey' and out.get('rise') is None:
        out.pop('rise')
    if name in {'leakage_vs_pulses', 'stark_shift_scan', 'sizzle_zz_scan', 'drag_calibration'}:
        out['include_reference'] = False
    for key in ARRAYS & out.keys():
        out[key] = np.asarray(out[key], dtype=int if key == 'n_list' else float)
    limit = specs[name].get('max_acquisitions')
    if limit is not None and acquisition_count(name, out) > limit:
        formula = '2 * len(amps) * len(delays)' if name == 'sizzle_zz_scan' else 'len(amps) * len(delays)'
        raise ValueError(f'Three-qubit chain {name}: {formula} must be <= {limit} acquisitions per request; split the scan across requests')
    return out

def acquisition_count(name, p):
    if name in _REGISTERED:
        return _REGISTERED[name][1](p)
    if name == 'resonator_spec':
        return len(p['freqs'])
    if name in {'gmm_readout', 'gmm3_readout'}:
        return 2 if name == 'gmm_readout' else 3
    axis = next((k for k in ('freqs', 'durations', 'delays', 'n_list', 'angles', 'betas', 'amps') if k in p))
    count = len(p[axis])
    if name in {'conditional_ramsey', 'cr_amp_scan', 'xy_plane_cnot_calibration', 'drag_calibration'}:
        count *= 2
    if name == 'cr_hamiltonian_tomography':
        count *= 6
    if name in {'spec_vs_power', 'stark_shift_scan', 'sizzle_zz_scan'}:
        count *= len(p['amps'])
        if name == 'sizzle_zz_scan':
            count *= 2
    return count
