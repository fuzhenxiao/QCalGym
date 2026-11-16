from copy import deepcopy
import json
import numpy as np
import matplotlib.pyplot as plt
from . import experiments
from .experiments.common import Result, new_fig
from .benchmark.catalog import register_experiment, catalog, validate, prepare, acquisition_count
from .benchmark.controls import scalar_bounds
from .acquisition_cache import reuse_acquisitions
KINDS = {'relaxation_series': ('t1', set(), set(), 5), 'resonator_power_map': ('resonator_spec', {'readout_amplitude'}, set(), 9), 'rabi_chevron': ('rabi_time', {'drive_freq'}, set(), 9), 'readout_power': ('gmm_readout', {'readout_amplitude'}, set(), 9), 'readout_duration': ('gmm_readout', {'readout_duration'}, set(), 9), 'ramsey_quadratures': ('ramsey', set(), {'virtual_detuning'}, 2), 'tone_quadratures': ('ramsey_with_tones', set(), {'virtual_detuning', 'spectator'}, 4), 'conditional_phase_scan': ('machine_probe', {'hold_amp', 'hold_freq', 'hold_phase', 'sizzle_duration'}, {'preparation', 'basis'}, 441), 'cnot_truth_scan': ('machine_probe', {'cr_amp', 'cr_phase', 'cr_duration'}, {'preparation'}, 441), 'cr_cancel_map': ('cr_hamiltonian_tomography', {'cr_cancel'}, set(), 2), 'readout_weights': ('state_readout', set(), {'preparation'}, 2), 'gate_frames': ('machine_probe', set(), {'preparation', 'basis'}, 8), 'state_tomography': ('machine_probe', set(), {'basis'}, 3), 'mixer_lo_map': ('mixer_spectrum', {'mixer_dc_i', 'mixer_dc_q'}, set(), 49), 'mixer_image_map': ('mixer_spectrum', {'mixer_gain', 'mixer_skew'}, set(), 49)}

def expand(p, env=None):
    name, fields, varying, limit = KINDS[p['kind']]
    rows = p['rows']
    if not 1 <= len(rows) <= limit:
        raise ValueError('Too many rows for standard scan family')
    specs = catalog(env.dev.n if env is not None else None)
    if name not in specs:
        raise ValueError('Scan primitive is unavailable')
    schema = specs[name]['parameters']
    out = []
    fixed = None
    control_keys = None
    for row in rows:
        args = deepcopy(row['parameters'])
        if 'shots' in args or 'n_avg' in args:
            raise ValueError('Use the shared scan shots parameter')
        args['n_avg' if name == 'resonator_spec' else 'shots'] = p['shots']
        validate(args, schema)
        args = {**{k: deepcopy(v['default']) for k, v in schema['properties'].items() if 'default' in v}, **args}
        if args.get('update_cal'):
            raise ValueError('Scan fits cannot install calibration')
        signature = json.dumps({k: v for k, v in args.items() if k not in varying}, sort_keys=True)
        if fixed is not None and signature != fixed:
            raise ValueError('Only declared scan axes may vary')
        fixed = signature
        controls = row.get('controls', {})
        if set(controls) - fields:
            raise ValueError('Control is not a scan axis for this family')
        if control_keys is not None and set(controls) != control_keys:
            raise ValueError('Scan axes must stay fixed')
        control_keys = set(controls)
        if p['kind'] == 'cnot_truth_scan' and len(controls) != 1:
            raise ValueError('One CNOT scan axis required')
        for field, value in controls.items():
            if field == 'cr_cancel':
                if not isinstance(value, dict) or set(value) != {'real', 'imag'} or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not np.isfinite(v)) or (abs(v) > 2) for v in value.values())):
                    raise ValueError('Invalid cancellation quadrature')
                continue
            lo, hi = (0.0, 2.0) if field == 'hold_amp' else scalar_bounds(4)[field]
            values = value if field == 'hold_amp' and isinstance(value, list) else [value]
            if field == 'hold_amp' and (not isinstance(value, list) or len(value) != 2):
                raise ValueError('Two hold amplitudes required')
            if any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not np.isfinite(v)) or (not lo <= v <= hi) for v in values)):
                raise ValueError('Scan control outside public bounds')
        index = p['index']
        key = tuple(index) if isinstance(index, list) else index
        expected = args.get('edge') if name == 'machine_probe' and args.get('gate') in ('cnot', 'cz', 'zz_idle') else args.get('k', args.get('source'))
        if expected is not None and expected != index:
            raise ValueError('Scan target must match primitive target')
        if env is not None:
            if isinstance(index, list):
                if tuple(index) not in env.dev.edges:
                    raise ValueError('Unknown scan edge')
            elif type(index) is not int or not 0 <= index < env.dev.n:
                raise ValueError('Unknown scan qubit')
            args = prepare(name, {k: v for k, v in row['parameters'].items()} | {'n_avg' if name == 'resonator_spec' else 'shots': p['shots']}, env)
        out.append((args, controls, key))
    return (name, out)

def count(p):
    name, rows = expand(p)
    total = sum((acquisition_count(name, args) for args, _, _ in rows))
    if total > 441:
        raise ValueError('At most 441 acquisitions per standard experiment; split the scan')
    return total

@reuse_acquisitions
def standard_scan(env, kind, index, rows, shots=1024):
    p = dict(kind=kind, index=index, rows=rows, shots=shots)
    count(p)
    name, expanded = expand(p, env)
    saved = env.cal.copy()
    results = []
    try:
        for args, controls, key in expanded:
            env.cal = saved.copy()
            for field, value in controls.items():
                getattr(env.cal, field)[key] = complex(value['real'], value['imag']) if field == 'cr_cancel' else deepcopy(value)
            env.sync_physics()
            r = getattr(experiments, name)(env, **args)
            results.append(dict(points=r.points, fit=r.fit, meta=r.meta))
            plt.close(r.figure)
    finally:
        env.cal = saved
        env.sync_physics()
    fig, ax = new_fig(kind, 'sample index', 'measured response')
    signal = {'resonator_spec': 's21_mag', 'rabi_time': 'p1', 'ramsey': 'p1', 't1': 'p1', 'ramsey_with_tones': 'p1', 'machine_probe': 'probabilities', 'mixer_spectrum': 'power'}.get(name)
    if signal and all((signal in r['points'] for r in results)):
        data = np.asarray([r['points'][signal] for r in results])
        ax.imshow(data, aspect='auto', origin='lower')
        ax.set_ylabel('scan row')
        ax.set_title(kind + ' / ' + signal)
    elif name == 'gmm_readout':
        ax.plot([r['fit'].get('params', {}).get('snr', float('nan')) for r in results], 'o-')
        ax.set_xlabel('scan row')
        ax.set_ylabel('fitted IQ SNR')
    elif name == 'cr_hamiltonian_tomography':
        for component in ('IX_GHz', 'IY_GHz', 'ZX_GHz', 'ZY_GHz'):
            ax.plot([r['fit'].get('params', {}).get(component, float('nan')) for r in results], 'o-', label=component)
        ax.set_xlabel('scan row')
        ax.set_ylabel('fitted Hamiltonian / GHz')
        ax.legend()
    else:
        for i, r in enumerate(results):
            for key, value in r['points'].items():
                values = np.asarray(value)
                if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
                    ax.plot(np.real(values), label=f'row {i}: {key}')
        ax.legend()
    return Result('standard_scan', dict(rows=results), {'status': 'finished', 'params': {}, 'model': 'per_row_fits'}, dict(kind=kind, primitive=name, shots=shots, temporary_controls_restored=True), fig)

def install(n):
    register_experiment('standard_scan', standard_scan, dict(description='One predeclared scan or tomography. No adaptive updates inside; per-row points/fits and all raw IQ retained. Same 441 acquisition cap as other experiments. Kinds: ' + ', '.join(KINDS), parameters=dict(type='object', additionalProperties=False, required=['kind', 'index', 'rows'], properties={'kind': dict(type='string', enum=list(KINDS)), 'index': dict(type=['integer', 'array'], minimum=0, maximum=n - 1, minItems=2, maxItems=2, items=dict(type='integer', minimum=0, maximum=n - 1)), 'rows': dict(type='array', minItems=1, maxItems=441, items=dict(type='object', additionalProperties=False, required=['parameters'], properties={'parameters': dict(type='object', additionalProperties=True), 'controls': dict(type='object', additionalProperties=True)})), 'shots': dict(type='integer', minimum=16, maximum=20000, default=1024)})), count)
