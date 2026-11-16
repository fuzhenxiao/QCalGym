from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
from .. import experiments
from ..env import Env
from .catalog import acquisition_count, catalog, prepare, validate, with_device_limits
from ..experiment_budget import BudgetExhausted
from .controls import scalar_bounds as control_bounds, control_catalog
from .data_access import VERSION, describe

def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, complex):
        return {'real': serial(value.real), 'imag': serial(value.imag)}
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items() if not str(k).startswith('_truth')}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def write_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')

@dataclass(frozen=True)
class Budget:
    max_calls: int | None = 40
    max_shots: int | None = 1000000
    max_acquisitions: int | None = 2000

    def __post_init__(self):
        if any((v is not None and (type(v) is not int or v <= 0) for v in asdict(self).values())):
            raise ValueError('Budget limits must be positive integers or None (unlimited)')

class RecordingEnv(Env):

    def begin_capture(self):
        self.raw_arrays, self.acquisitions = ({}, [])

    def _sample_simulation(self, pops, t_end, shots, qubits, name):
        iq = super()._sample_simulation(pops, t_end, shots, qubits, name)
        index = len(self.acquisitions)
        keys = {}
        for k, array in iq.items():
            key = f'acq_{index:04d}_q{k}_iq'
            self.raw_arrays[key] = array.copy()
            keys[str(k)] = key
        self.acquisitions.append({'index': index, 'name': name, 'shots': shots, 'iq_arrays': keys, 'duration_ns': self.ledger.log[-1]['duration_ns']})
        return iq

    def measure_traces(self, seq, shots, qubits=None, name='readout_trace', rho0=None):
        traces = super().measure_traces(seq, shots, qubits, name, rho0)
        index, keys = (len(self.acquisitions), {})
        for k, array in traces.items():
            key = f'acq_{index:04d}_q{k}_time_iq'
            self.raw_arrays[key] = array.copy()
            keys[str(k)] = key
        self.acquisitions.append({'index': index, 'name': name, 'shots': shots, 'time_iq_arrays': keys, 'duration_ns': self.ledger.log[-1]['duration_ns']})
        return traces

class Session:

    def __init__(self, directory, *, env, seed=0, budget=Budget(), objective=None, experiment_catalog=None, allowed_updates=None, experiment_runtime=True):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        base = env
        if 'flux_coupler' in base.dev.physics:
            raise ValueError('Use backend.coupler_session.FluxCouplerSession for flux-coupler hardware; legacy experiment catalog is not its control contract')
        self.env = RecordingEnv(deepcopy(base.dev), base.cal.copy(), seed=seed, solver_options={**base.sim.opts, 'exact_flat': base.sim.exact_flat, 'exact_idle': base.sim.exact_idle, 'noise_order': base.sim.noise_order, 'noise_max_order': base.sim.noise_max_order, 'noise_tolerance': base.sim.noise_tolerance}, solver_mode=base.sim.mode)
        from ..parametric import configure_scoring
        configure_scoring(self.env, getattr(base.sim, 'channel_cache_path', None))
        if hasattr(base, '_channel_score_cache'):
            self.env._channel_score_cache = deepcopy(base._channel_score_cache)
        self.env.begin_capture()
        self.budget = budget
        self.experiment_catalog = with_device_limits(catalog(self.env.dev.n) if experiment_catalog is None else experiment_catalog, self.env.dev.n, self.env.dev.topology)
        if self.env.dev.n <= 2:
            from ..hardware_probes import install
            self.experiment_catalog.update(install(self.env.dev))
        from ..parametric import enabled
        if enabled(self.env.dev):
            from ..parametric_probe import configure
            self.experiment_catalog = configure(self.experiment_catalog)
        from ..instruments import install as install_instruments
        from ..hardware import apparatus_eligible
        if not apparatus_eligible(self.env.dev):
            self.experiment_catalog.pop('mixer_spectrum', None)
        self.experiment_catalog.update(install_instruments(self.env.dev))
        self.allowed_updates = None if allowed_updates is None else set(allowed_updates)
        self.objective = deepcopy(objective) if objective is not None else {'instruction': 'Nominal apparatus integration session; no scored error case has been installed.'}
        self.calls, self.revision = (0, 0)
        self.submission = None
        self.receipts = {}
        self.lock = threading.RLock()
        self._artifact_lock = threading.Lock()
        self._public_artifacts = set()
        root = Path(__file__).resolve().parents[1]
        from ..sim_client import enabled, backend_mode, version
        simulation_backend = {'mode': backend_mode(), 'version': version(), 'idle': 'full_lindblad_exponential', 'pulse': 'qutip_mesolve'} if enabled(self.env.sim) else {'mode': 'off'}
        self.experiment_sim = None
        setup_started = time.monotonic()
        from ..hardware import dynamics_enabled
        if experiment_runtime and self.env.sim.mode == 'qutip' and (not enabled(self.env.sim)) and (not dynamics_enabled(self.env.dev)):
            from ..sim_engine import Engine
            options = {**self.env.sim.opts, 'method': 'dop853', 'exact_flat': True}
            default_workers = 2 if self.env.dev.n == 3 and self.env.sim.D >= 64 else 1
            import __main__
            if not Path(getattr(__main__, '__file__', '')).is_file():
                default_workers = 1
            workers = int(os.environ.get('QCAL_ACQUISITION_WORKERS', str(default_workers)))
            if not 1 <= workers <= 8:
                raise ValueError('QCAL_ACQUISITION_WORKERS must be between 1 and 8')
            if workers > 1:
                from ..acquisition_pool import ParallelEngine
                self.experiment_sim = ParallelEngine(self.env.dev, options, workers=workers)
            else:
                self.experiment_sim = Engine(self.env.dev, options)
            self.experiment_sim.dressed()
            if self.experiment_sim.D <= 81:
                self.experiment_sim._build_idle_blocks()
        experiment_setup_seconds = time.monotonic() - setup_started
        write_json(self.directory / 'host_manifest.json', {'data_api_version': VERSION, 'seed': seed, 'budget': asdict(budget), 'solver_mode': self.env.sim.mode, 'experiment_catalog': self.experiment_catalog, 'allowed_updates': sorted(self.allowed_updates) if self.allowed_updates is not None else None, 'solver_options': {**self.env.sim.opts, 'exact_flat': self.env.sim.exact_flat, 'exact_idle': self.env.sim.exact_idle, 'noise_order': self.env.sim.noise_order, 'noise_max_order': self.env.sim.noise_max_order, 'noise_tolerance': self.env.sim.noise_tolerance}, 'simulation_backend': simulation_backend, 'optional_hardware_dynamics': 'full_sequence_lindblad' if dynamics_enabled(self.env.dev) else None, 'experiment_runtime': {'mode': 'full_lindblad_cached' if self.experiment_sim is not None else 'inherited', 'setup_seconds': experiment_setup_seconds, 'workers': getattr(self.experiment_sim, 'workers', 1), 'solver_options': self.experiment_sim.opts if self.experiment_sim is not None else None, 'exact_idle': self.experiment_sim.exact_idle if self.experiment_sim is not None else None, 'exact_flat': self.experiment_sim.exact_flat if self.experiment_sim is not None else None}, 'device': asdict(self.env.dev), 'initial_calibration': asdict(self.env.cal), 'objective': self.objective, 'implementation': 'qcalibench_v4'})

    def status(self):
        value = {'calibration': serial(asdict(self.env.cal)), 'revision': self.revision, 'budget': asdict(self.budget), 'used': {'calls': self.calls, **self.env.budget()}, 'objective': self.objective, 'submitted': self.submission is not None, 'n_qubits': self.env.dev.n, 'edges': self.env.dev.edges}
        from ..hardware import public_apparatus, apparatus_eligible
        if self.env.dev.physics or apparatus_eligible(self.env.dev):
            value['apparatus'] = public_apparatus(self.env.dev)
        return value

    def _update(self, updates):
        if not isinstance(updates, list) or not 1 <= len(updates) <= 64:
            raise ValueError('updates must be a nonempty list with at most 64 entries')
        candidate = self.env.cal.copy()
        scalar_bounds = control_bounds(self.env.dev.n)
        from ..parametric import enabled, control_bounds as coupler_bounds, obsolete_control
        parametric = enabled(self.env.dev)
        if parametric:
            scalar_bounds.update(coupler_bounds())
        pair_fields = {name for name in asdict(candidate) if name.startswith(('cr_', 'ecr_', 'sizzle_', 'post_gate_', 'hold_', 'coupler_'))}
        for update in updates:
            if not isinstance(update, dict) or set(update) != {'field', 'index', 'value'}:
                raise ValueError('Each update needs exactly field, index, value')
            field, index, value = (update[k] for k in ('field', 'index', 'value'))
            if self.allowed_updates is not None and field not in self.allowed_updates:
                raise ValueError(f'Calibration field {field} is fixed in this episode')
            if field not in asdict(candidate):
                raise ValueError('Unknown calibration field')
            if parametric and obsolete_control(field):
                raise ValueError('This control is not part of the parametric three-qubit apparatus')
            if field in pair_fields:
                if not isinstance(index, list) or len(index) != 2 or any((type(k) is not int for k in index)):
                    raise ValueError('Pair index must be [control, target]')
                if not any((set(index) == set(e) for e in self.env.dev.edges)):
                    raise ValueError('Pair must be a device edge')
                index = tuple(index)
            elif type(index) is not int or not 0 <= index < self.env.dev.n:
                raise ValueError('Index must be a device qubit')
            if field == 'idle_active' and index != 0:
                raise ValueError('idle_active has only global key 0')
            if field in scalar_bounds:
                lo, hi = scalar_bounds[field]
                validate(value, {'type': 'integer' if field in {'drive_line', 'hold_enabled', 'idle_active', 'idle_tone_smooth', 'sizzle_smooth', 'hold_smooth', 'ecr_walsh', 'dd_pulses'} else 'number', 'minimum': lo, 'maximum': hi})
            elif field == 'dd_sequence':
                validate(value, {'type': 'string', 'enum': ['off', 'cpmg', 'xy4', 'xy8']})
            elif field == 'readout_weights':
                validate(value, {'type': 'array', 'minItems': 1, 'maxItems': 256, 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number', 'minimum': -100, 'maximum': 100}}})
            elif field in {'sizzle_network_amp', 'sizzle_network_phase', 'post_gate_network_phase'}:
                lo, hi = (0, 2) if field == 'sizzle_network_amp' else (-20, 20)
                validate(value, {'type': 'array', 'minItems': self.env.dev.n, 'maxItems': self.env.dev.n, 'items': {'type': 'number', 'minimum': lo, 'maximum': hi}})
            elif field in {'cr_cancel', 'cr_rotary'}:
                value = self._complex(value)
            elif field in {'sizzle_amp', 'post_gate_phase', 'hold_amp', 'hold_frame'}:
                bound = 2 if field in {'sizzle_amp', 'hold_amp'} else 20
                validate(value, {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number', 'minimum': -bound, 'maximum': bound}})
                value = tuple(value)
            elif field == 'source_mix':
                if not isinstance(value, list) or len(value) != self.env.dev.n:
                    raise ValueError('source_mix needs one [real,imag] per physical source')
                validate(value, {'type': 'array', 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number', 'minimum': -2, 'maximum': 2}}})
            elif field == 'iq_means3':
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError('iq_means3 needs three complex {real, imag} values')
                value = [self._complex(v, 2000) for v in value]
            getattr(candidate, field)[index] = value
        for edge, rise in candidate.cr_rise.items():
            if 2 * rise > candidate.cr_duration[edge]:
                raise ValueError('CR duration must be at least twice its rise time')
        for edge, rise in candidate.sizzle_rise.items():
            if 2 * rise > candidate.sizzle_duration[edge]:
                raise ValueError('siZZle duration must be at least twice its rise time')
        from ..hardware import validate_controls
        validate_controls(self.env.dev, candidate)
        self.env.cal = candidate
        self.env.sync_physics()

    @staticmethod
    def _complex(value, bound=2):
        validate(value, {'type': 'object', 'required': ['real', 'imag'], 'additionalProperties': False, 'properties': {k: {'type': 'number', 'minimum': -bound, 'maximum': bound} for k in ('real', 'imag')}})
        return complex(value['real'], value['imag'])

    def dispatch(self, action, params=None, request_id=None):
        with self.lock:
            started = time.time()
            try:
                response = self._dispatch(action, params or {}, request_id)
            except Exception as exc:
                self._log_api_call(action, params, request_id, started, {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
                raise
            self._log_api_call(action, params, request_id, started, response)
            return response

    def _log_api_call(self, action, params, request_id, started, response):
        with (self.directory / 'api_calls.jsonl').open('a') as f:
            f.write(json.dumps(serial(dict(action=action, params=params or {}, request_id=request_id, started_at=started, finished_at=time.time(), response=response)), allow_nan=False) + '\n')

    def _dispatch(self, action, params, request_id):
        if action == 'catalog':
            return {'status': 'success', 'api_version': VERSION, 'experiments': self.experiment_catalog, 'state': self.status()}
        if action == 'controls':
            return {'status': 'success', **control_catalog(self.env.dev.n, self.env.dev.edges, self.allowed_updates, self.env.cal, self.env.dev)}
        if action == 'status':
            return {'status': 'success', 'state': self.status()}
        if action == 'get_data':
            return self._get_data(**params)
        if action == 'check_raw_data':
            return self._check_raw_data(**params)
        if not isinstance(request_id, str) or not re.fullmatch('[A-Za-z0-9_-]{1,80}', request_id):
            raise ValueError('A request_id containing 1-80 letters, digits, _ or - is required')
        fingerprint = json.dumps([action, params], sort_keys=True, allow_nan=False)
        if request_id in self.receipts:
            old, response = self.receipts[request_id]
            if old != fingerprint:
                raise ValueError('request_id already used with different input')
            return deepcopy(response)
        if self.submission is not None:
            return {'status': 'failed', 'error_type': 'episode_closed', 'error': 'Episode already submitted'}
        if self.budget.max_calls is not None and self.calls >= self.budget.max_calls and (action != 'submit'):
            return {'status': 'failed', 'error_type': 'budget_exhausted', 'state': self.status()}
        if action != 'submit':
            self.calls += 1
        folder = self.directory / f'{len(self.receipts) + 1:04d}_{request_id}'
        folder.mkdir()
        self.env.begin_capture()
        before = self.env.cal.copy()
        ledger_before = {key: getattr(self.env.ledger, key) for key in ('shots', 'sim_time_ns')}
        started = time.monotonic()
        write_json(folder / 'request.json', {'action': action, 'params': params, 'request_id': request_id})
        open_figures = set(plt.get_fignums())
        try:
            if action == 'run_experiment':
                if set(params) != {'name', 'parameters'}:
                    raise ValueError('run_experiment requires name and parameters')
                name = params['name']
                if name not in self.experiment_catalog:
                    raise ValueError('Experiment is unavailable in this episode')
                validate(params['parameters'], self.experiment_catalog[name]['parameters'])
                if self.allowed_updates is not None and params['parameters'].get('update_cal'):
                    raise ValueError('Experiment-side calibration updates are disabled in this episode')
                kwargs = prepare(name, params['parameters'], self.env)
                count = acquisition_count(name, kwargs)
                shots = count * kwargs.get('shots', kwargs.get('n_avg', 200))
                if self.budget.max_shots is not None and self.env.ledger.shots + shots > self.budget.max_shots:
                    raise BudgetExhausted('shot budget would be exceeded; no data acquired')
                if self.budget.max_acquisitions is not None and sum((v[1].get('cost', {}).get('acquisitions', 0) for v in self.receipts.values())) + count > self.budget.max_acquisitions:
                    raise BudgetExhausted('acquisition budget would be exceeded; no data acquired')
                from ..hardware import apparatus_controls_active
                if apparatus_controls_active(self.env.cal):
                    self._fast_engine_disabled = True
                use_fast = self.experiment_sim is not None and (not getattr(self, '_fast_engine_disabled', False))
                scoring_sim = self.env.sim
                try:
                    if use_fast:
                        self.env.sim = self.experiment_sim
                    self.env.sync_physics()
                    result = getattr(experiments, name)(self.env, **kwargs)
                finally:
                    if use_fast:
                        self.env.sim = scoring_sim
                points = serial(result.points)
                if name == 'resonator_spec':
                    for key, value in result.points.items():
                        self.env.raw_arrays[key] = np.asarray(value)
                figures = []
                if getattr(self, 'save_figures', True):
                    result.figure.savefig(folder / 'figure.png', dpi=150, bbox_inches='tight')
                    figures = [{'format': 'png', 'artifact': f'{folder.name}/figure.png'}]
                response = {'status': 'success', 'experiment': name, 'points': points, 'fit': serial(result.fit), 'meta': serial(result.meta), 'figures': figures, 'cost': {'acquisitions': count}}
            elif action == 'update_calibration':
                if set(params) != {'updates'}:
                    raise ValueError('update_calibration requires updates')
                self._update(params['updates'])
                response = {'status': 'success'}
            elif action == 'submit':
                validate(params, {'type': 'object', 'required': ['report'], 'additionalProperties': False, 'properties': {'report': {'type': 'string', 'maxLength': 16000}, 'claims': {'type': 'object', 'additionalProperties': True}}})
                self.submission = deepcopy(params)
                response = {'status': 'success', 'submission_received': True, 'note': 'Receipt is not a pass verdict; scoring is performed by the author separately.'}
            else:
                raise ValueError('Unknown action')
        except Exception as exc:
            self.env.cal = before
            self.env.sync_physics()
            response = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
        finally:
            for num in set(plt.get_fignums()) - open_figures:
                plt.close(num)
        if serial(asdict(before)) != serial(asdict(self.env.cal)):
            self.revision += 1
        actual_shots = self.env.ledger.shots - ledger_before['shots']
        response.setdefault('cost', {}).update(shots=actual_shots, sim_time_ns=self.env.ledger.sim_time_ns - ledger_before['sim_time_ns'], host_seconds=time.monotonic() - started)
        response['cost'].setdefault('acquisitions', len(self.env.acquisitions))
        if action == 'run_experiment' and params.get('name') == 'standard_scan':
            response['cost']['acquisitions'] = actual_shots // params['parameters'].get('shots', 1024)
        np.savez_compressed(folder / 'raw.npz', **self.env.raw_arrays)
        response['raw_data'] = {'artifact': f'{folder.name}/raw.npz', 'kind': 'averaged_s21_trace' if action == 'run_experiment' and params.get('name') == 'resonator_spec' else 'time_bin_iq' if any(('time_iq_arrays' in x for x in self.env.acquisitions)) else 'single_shot_iq', 'acquisitions': self.env.acquisitions, 'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in self.env.raw_arrays.items()}}
        response.update(request_id=request_id, state=self.status())
        response['response_artifact'] = f'{folder.name}/response.json'
        response = serial(response)
        write_json(folder / 'response.json', response)
        write_json(folder / 'calibration_after.json', asdict(self.env.cal))
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'request_id': request_id, 'action': action, 'params': params, 'response': response}, allow_nan=False) + '\n')
        self.receipts[request_id] = (fingerprint, deepcopy(response))
        with self._artifact_lock:
            self._public_artifacts.add(response['raw_data']['artifact'])
            self._public_artifacts.add(response['response_artifact'])
            self._public_artifacts.update((f['artifact'] for f in response.get('figures', [])))
        return response

    def _check_raw_data(self, request_id, section='manifest', array=None, start=0, stop=100):
        if request_id not in self.receipts:
            raise ValueError('Unknown acquisition request')
        if type(start) is not int or type(stop) is not int or (not 0 <= start < stop <= start + 2000):
            raise ValueError('Slice must contain 1-2000 entries')
        response = self.receipts[request_id][1]
        raw = response.get('raw_data', {})
        if section == 'raw':
            return self._get_data(request_id, array, start, stop)
        if section == 'manifest':
            entries = list(raw.get('arrays', {}).items())
            acquisitions = raw.get('acquisitions', [])
            return {'status': 'success', 'request_id': request_id, 'point_arrays': {k: describe(v) for k, v in response.get('points', {}).items()}, 'raw_arrays': dict(entries[start:stop]), 'raw_array_count': len(entries), 'acquisitions': deepcopy(acquisitions[start:stop]), 'acquisition_count': len(acquisitions), 'start': start, 'stop': stop, 'next_start': stop if stop < max(len(entries), len(acquisitions)) else None, 'raw_artifact': raw.get('artifact'), 'response_artifact': response.get('response_artifact')}
        if section == 'points':
            points = response.get('points', {})
            if array not in points:
                raise ValueError("Unknown points array; use section='manifest' to discover names")
            values = points[array]
            if not isinstance(values, list):
                raise ValueError('Selected point value is not an array; download the public response')
            return {'status': 'success', 'array': array, 'start': start, 'length': len(values), 'values': deepcopy(values[start:stop])}
        if section in {'fit', 'meta', 'state', 'response'}:
            return {'status': 'success', 'request_id': request_id, section: deepcopy(response if section == 'response' else response.get(section, {}))}
        raise ValueError('section must be manifest, points, raw, fit, meta, state or response')

    def _get_data(self, request_id, array, start=0, stop=2000):
        if request_id not in self.receipts:
            raise ValueError('Unknown acquisition request')
        response = self.receipts[request_id][1]
        if array not in response['raw_data']['arrays']:
            raise ValueError('Unknown raw array')
        if type(start) is not int or type(stop) is not int or (not 0 <= start < stop <= start + 2000):
            raise ValueError('Slice must contain 1-2000 samples')
        with np.load(self.directory / response['raw_data']['artifact'], allow_pickle=False) as archive:
            value = archive[array][start:stop]
        return {'status': 'success', 'array': array, 'start': start, 'values': serial(value)}

    def artifact(self, relative):
        with self._artifact_lock:
            if relative not in self._public_artifacts:
                raise ValueError('Unknown public artifact')
        return self.directory / relative

    def evaluate(self, author_scorer):
        if self.submission is None:
            raise ValueError('No submission to score')
        result = author_scorer(deepcopy(self.env.dev), self.env.cal.copy(), deepcopy(self.submission))
        write_json(self.directory / 'author_score.json', result)
        return result
