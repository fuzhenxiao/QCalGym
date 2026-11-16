from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import threading
import numpy as np
from .session import Session, INSTRUCTION
from .benchmark.session import RecordingEnv, Budget, serial, write_json
from .benchmark.catalog import register_experiment, validate
from .coupler_session import FluxCouplerSession
from .flux_scan import FIELDS
from .experiments.common import Result, new_fig
BASE_BOUNDS = {'drive_freq': (0.1, 15.0), 'readout_freq': (0.1, 15.0), 'pi_amp': (0.0, 2.0), 'pi_duration': (1.0, 2000.0), 'drag': (-5.0, 5.0), 'anharm': (-2.0, -0.001), 'iq_angle': (-20.0, 20.0), 'iq_threshold': (-2000.0, 2000.0), 'readout_amplitude': (0.0, 1000.0), 'readout_duration': (1.0, 20000.0)}

def number(lo, hi):
    return dict(type='number', minimum=lo, maximum=hi)

def array(item, n):
    return dict(type='array', items=item, minItems=n, maxItems=n)

def control_schema(dev):
    ports = [k for k, f in enumerate(dev.physics['flux_coupler']['qubit_flux'] + [dev.physics['flux_coupler']['coupler_flux']]) if f]
    fields = {k: dict(value=number(*v), index={'type': 'integer', 'enum': [0, 1]}) for k, v in BASE_BOUNDS.items()}
    descriptions = {'drive_freq': 'Microwave carrier GHz; independent of physical flux bias.', 'pi_amp': 'X180 Gaussian amplitude; X90 has half amplitude at the same pi_duration.', 'pi_duration': 'Gaussian gate duration ns; does not rescale pi_amp.', 'anharm': 'Estimated f12-f01 for DRAG, GHz.', 'readout_amplitude': 'Readout voltage; power is proportional to its square.', 'readout_duration': 'IQ integration window ns.', 'iq_angle': 'Classifier angle rad; assigned 1 when Re(IQ exp(-i angle)) > iq_threshold.'}
    for k, v in fields.items():
        v['description'] = descriptions.get(k, k.replace('_', ' '))

    def add(k, value, indices, description):
        fields[k] = dict(value=value, index={'type': 'integer', 'enum': indices}, description=description)
    if 0 in ports:
        add('flux_bias', number(-0.45, 0.45), [0, 1], 'Independent data-qubit DC flux, Phi0. Recalibrate microwave frequency after changing it.')
    add('coupler_flux_bias', number(-0.45, 0.45), [0], 'Coupler DC flux, Phi0.')
    add('flux_mix', array(array(number(-10, 10), 3), 3), [0], '3x3 digital flux crosstalk compensation. Only installed Z output ports may be used.')
    add('flux_inverse_tau', number(0, 20000), ports, 'Inverse flux-filter time constant per Z port, ns; zero disables.')
    add('flux_inverse_tail', number(1e-06, 1), ports, 'Fraction of step response in the inverted exponential tail.')
    native = dict(type='object', additionalProperties=False, properties=dict(mode={'type': 'string', 'enum': ['baseband_cz', 'parametric_cz']}, duration=number(1, 20000), amplitudes=array(number(-0.45, 0.45), 3), rise_fractions=array(number(0, 0.5), 3), frequency=number(0, 5), phase=number(-100, 100), settle=number(0, 20000), frames=array(number(-100, 100), 2)))
    native['required'] = list(native['properties'])
    add('native_flux_gate', native, [0], 'Installed CZ: three port amplitudes/rise fractions, carrier GHz, duration/settle ns and two logical frame corrections rad. Fixed-frequency data qubits require amplitudes[0:2]=0. Baseband frequency=0; parametric frequency>0.')
    return dict(fields=fields, max_updates_per_call=64, flux_ports=ports, constraints=['Updates are atomic; only calibration controls change.', 'Flux pulses must remain within physical flux ranges after mixing/filtering.', 'Scans restore controls afterwards; explicitly update calibration to install a result.', 'Cold start has empty controls. Set approximate drive/readout frequencies before preparing excited states.'])

def normalized(request):
    result = deepcopy(request)
    controls = result.get('controls', {})
    for field, value in controls.get('readout', {}).items():
        controls['readout'][field] = {int(k): v for k, v in value.items()}
    for field in ('qubit_bias', 'inverse_tau', 'inverse_tail'):
        if field in controls.get('flux', {}):
            controls['flux'][field] = {int(k): v for k, v in controls['flux'][field].items()}
    return result

def measured_result(env, iqs, name, requests):
    points = {'row': list(range(len(iqs)))}
    fig, ax = new_fig(name, xlabel='Request row', ylabel='Mean measured IQ')
    for q in range(2):
        means = np.array([iq[q].mean() for iq in iqs])
        points[f'q{q}_I'] = means.real.tolist()
        points[f'q{q}_Q'] = means.imag.tolist()
        if q in env.cal.iq_angle and q in env.cal.iq_threshold:
            points[f'q{q}_assigned_P1'] = [float(np.mean((iq[q] * np.exp(-1j * env.cal.iq_angle[q])).real > env.cal.iq_threshold[q])) for iq in iqs]
        ax.plot(points['row'], means.real, 'o-', label=f'Q{q} I')
        ax.plot(points['row'], means.imag, 'o--', label=f'Q{q} Q')
    ax.legend()
    return Result(name, points, {}, dict(requests=requests, observable='Measured IQ including finite preparation/analysis and shot noise'), fig)

def flux_iq_scan(env, kind, requests, shots=1024):
    rows = [normalized(r) for r in requests]
    results = env._public_flux_session.run_scan(kind, rows, shots=shots)
    return measured_result(env, results, kind, requests)

def flux_iq_acquire(env, blocks, preparation=('0', '0'), basis=('Z', 'Z'), analysis_phases=(0.0, 0.0), shots=1024):
    request = dict(blocks=blocks, preparation=preparation, basis=basis, analysis_phases=analysis_phases, shots=shots)
    result = env._public_flux_session.acquire(**request)
    return measured_result(env, [result], 'Flux IQ acquisition', [request])

def install_experiments():
    states = array({'type': 'string', 'enum': ['0', '1', '2', '+', '+i']}, 2)
    bases = array({'type': 'string', 'enum': ['X', 'Y', 'Z']}, 2)
    shots = dict(type='integer', minimum=16, maximum=20000, default=1024)
    specs = {}

    def add(name, fn, description, properties, required, counter):
        spec = dict(description=description, parameters=dict(type='object', properties=properties, required=required, additionalProperties=False))
        register_experiment(name, fn, spec, counter)
        specs[name] = spec
    block_description = 'Blocks are waits in ns, {"xy":[{"qubit":0,"angle":1.570796,"phase":0,"amp":0.2,"duration":40,"drag":0,"freq":5}]} (XY keys except qubit optional), {"duration":100,"pulses":[{"port":2,"amp":0.1,"duration":100,"rise":20,"freq":0,"phase":0}]}, or {"gate":"cz","repetitions":1}. Z ports 0/1=data qubits when tunable, 2=coupler. XY and Z blocks run sequentially; waits include line tails.'
    add('flux_iq_acquire', flux_iq_acquire, 'Acquire one two-channel IQ experiment. ' + block_description, dict(blocks={'type': 'array', 'maxItems': 1024, 'items': {'type': ['number', 'object'], 'additionalProperties': True}}, preparation={**states, 'default': ['0', '0']}, basis={**bases, 'default': ['Z', 'Z']}, analysis_phases={**array(number(-100, 100), 2), 'default': [0.0, 0.0]}, shots=shots), ['blocks'], lambda p: 1)
    descriptions = 'A predefined scan costs one standard experiment, 1..441 acquisitions. Each request contains blocks, preparation, basis, analysis_phases, controls as needed. controls groups: readout={field:{"0":value,"1":value}}, flux={qubit_bias:{"0":value},coupler_bias:value,mix:3x3,inverse_tau:{"2":value},inverse_tail:{"2":value}}, native={gate_field:value}. Only the listed axes for the chosen family may vary, consistently across rows; no controls are installed by scans. qubit_spectrum/rabi/drag use XY blocks; ramsey/coherence use waits with + preparation and X/Y analysis; flux_ramsey uses Z blocks/waits; native_phase uses one CZ block and varies one gate field per scan. Allowed axes: ' + str({k: {g: sorted(v) for g, v in groups.items()} for k, groups in FIELDS.items()}) + '. ' + block_description
    add('flux_iq_scan', flux_iq_scan, descriptions, dict(kind={'type': 'string', 'enum': list(FIELDS)}, requests={'type': 'array', 'minItems': 1, 'maxItems': 441, 'items': {'type': 'object', 'additionalProperties': True}}, shots=shots), ['kind', 'requests'], lambda p: len(p['requests']))
    return specs

class FluxBenchmarkSession(Session):

    def __init__(self, directory, task, env, initial, experiment_limit=100, seed=0, budget=Budget(), **kwargs):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.task = task
        self.experiment_limit = experiment_limit
        self.experiment_count = 0
        self.initial_performance = initial
        self.env = RecordingEnv(deepcopy(env.dev), env.cal.copy(), seed=seed, solver_options=env.sim.opts, solver_mode=env.sim.mode)
        engine = FluxCouplerSession(env.dev, cal=env.cal, seed=seed, max_acquisitions=budget.max_acquisitions, max_shots=budget.max_shots, solver_options=env.sim.opts, solver_mode=env.sim.mode, max_experiments=None)
        engine._env = self.env
        self.env._public_flux_session = engine
        self.env.begin_capture()
        self.budget = budget
        self.experiment_catalog = install_experiments()
        self.allowed_updates = set(control_schema(env.dev)['fields'])
        from .session import public_objective
        self.objective = public_objective(initial, task.family)
        self.calls = self.revision = 0
        self.submission = None
        self.receipts = {}
        self.lock = threading.RLock()
        self._artifact_lock = threading.Lock()
        self._public_artifacts = set()
        self.experiment_sim = None
        write_json(self.directory / 'host_manifest.json', dict(device=asdict(env.dev), initial_calibration=asdict(env.cal), objective=self.objective, solver_options=env.sim.opts, experiment_catalog=self.experiment_catalog, implementation='FluxCouplerSession public bridge', budget=asdict(budget), seed=seed))

    def _dispatch(self, action, params, request_id):
        if action == 'controls':
            return {'status': 'success', **control_schema(self.env.dev)}
        return super()._dispatch(action, params, request_id)

    def _update(self, updates):
        schema = control_schema(self.env.dev)['fields']
        if not isinstance(updates, list) or not 1 <= len(updates) <= 64:
            raise ValueError('Expected 1..64 updates')
        candidate = self.env.cal.copy()
        for update in updates:
            if not isinstance(update, dict) or set(update) != {'field', 'index', 'value'}:
                raise ValueError('Update needs field, index, value')
            field, index, value = (update[k] for k in ('field', 'index', 'value'))
            if field not in schema:
                raise ValueError('Unavailable flux calibration control')
            validate(index, schema[field]['index'])
            validate(value, schema[field]['value'])
            getattr(candidate, field)[index] = deepcopy(value)
        check = FluxCouplerSession(self.env.dev, cal=candidate, solver_options=self.env.sim.opts, solver_mode=self.env.sim.mode)
        check.set_readout_drive_controls(**{k: deepcopy(getattr(candidate, k)) for k in BASE_BOUNDS})
        if candidate.native_flux_gate:
            check.set_native_gate(**candidate.native_flux_gate[0])
        self.env.cal = check._env.cal.copy()
        self.env.sync_physics()
