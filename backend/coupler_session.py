from copy import deepcopy
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .env import Env
from .device import CalibrationStore
from .flux_coupler import FluxPulse, FluxSegment
from .hardware import public_apparatus
from .hardware_probes import prepare_product, analysis
from .core import Segment
from .flux_gates import validate_gate, compile_gate

class FluxCouplerSession:

    def __init__(self, dev, *, cal=None, seed=0, directory=None, max_acquisitions=100, max_shots=2000000, solver_options=None, solver_mode='fast', max_experiments=None):
        if any((v is not None and (type(v) is not int or v < 1) for v in (max_acquisitions, max_shots, max_experiments))):
            raise ValueError('Budgets must be positive integers or None (unlimited)')
        self._env = Env(deepcopy(dev), deepcopy(cal) if cal is not None else CalibrationStore(), seed=seed, solver_options=solver_options, solver_mode=solver_mode)
        if 'flux_coupler' not in dev.physics:
            raise ValueError('Flux-coupler device required')
        self._max_acquisitions = max_acquisitions
        self._max_shots = max_shots
        self._directory = Path(directory) if directory else None
        if self._directory:
            self._directory.mkdir(parents=True, exist_ok=False)
        self._observation_cache = {}
        self._native_validation_cache = OrderedDict()
        self.experiment_count = 0
        self._max_experiments = max_experiments
        self._in_scan = False

    def _record_update(self, action, values):
        if self._directory:
            with (self._directory / 'updates.jsonl').open('a') as f:
                f.write(json.dumps(dict(action=action, values=values, used=self._env.budget())) + '\n')

    def status(self):
        return {'apparatus': public_apparatus(self._env.dev), 'calibration': {k: deepcopy(getattr(self._env.cal, k)) for k in ('drive_freq', 'readout_freq', 'pi_amp', 'pi_duration', 'drag', 'anharm', 'iq_angle', 'iq_threshold', 'readout_amplitude', 'readout_duration')}, 'used': {**self._env.budget(), 'experiment_calls': self.experiment_count}, 'experiment_limit': self._max_experiments, 'limits': {'acquisitions': self._max_acquisitions, 'shots': self._max_shots}, 'flux_controls': {k: deepcopy(getattr(self._env.cal, k)) for k in ('flux_bias', 'coupler_flux_bias', 'flux_mix', 'flux_inverse_tau', 'flux_inverse_tail')}, 'native_gate': deepcopy(self._env.cal.native_flux_gate.get(0))}

    def set_flux_controls(self, **updates):
        self._env.set_flux_controls(**updates)
        self._record_update('set_flux_controls', updates)
        return self.status()

    def set_native_gate(self, **updates):
        candidate = {**self._env.cal.native_flux_gate.get(0, {}), **deepcopy(updates)}
        candidate = validate_gate(self._env.dev, candidate)
        cal = self._env.cal.copy()
        cal.native_flux_gate[0] = candidate
        from .flux_coupler import flux_trace
        old = self._env.cal
        try:
            self._env.cal = cal
            controls = {k: getattr(cal, k) for k in ('flux_bias', 'coupler_flux_bias', 'flux_mix', 'flux_inverse_tau', 'flux_inverse_tail')}
            key = repr((self._env.dev.to_json(), controls, candidate, self._env.sim.opts['flux_dt']))
            if key not in self._native_validation_cache:
                seq, _ = compile_gate(self._env)
                duration, pulses, _, _ = self._env.sim._schedule(seq)
                flux_trace(self._env.dev, cal, pulses, duration, self._env.sim.opts['flux_dt'])
                self._native_validation_cache[key] = True
                if len(self._native_validation_cache) > 128:
                    self._native_validation_cache.popitem(last=False)
            self._native_validation_cache.move_to_end(key)
        except Exception:
            self._env.cal = old
            raise
        self._record_update('set_native_gate', updates)
        return self.status()

    def set_readout_drive_controls(self, **updates):
        bounds = {'drive_freq': (0.1, 15.0), 'readout_freq': (0.1, 15.0), 'pi_amp': (0.0, 2.0), 'pi_duration': (1.0, 2000.0), 'drag': (-5.0, 5.0), 'anharm': (-2.0, -0.001), 'iq_angle': (-20.0, 20.0), 'iq_threshold': (-2000.0, 2000.0), 'readout_amplitude': (0.0, 1000.0), 'readout_duration': (1.0, 20000.0)}
        candidate = self._env.cal.copy()
        from .flux_coupler import finite
        for key, value in updates.items():
            if key not in bounds or not isinstance(value, dict) or set(value) - {0, 1}:
                raise ValueError('Unknown control or qubit')
            for k, v in value.items():
                finite(v, key, *bounds[key])
            getattr(candidate, key).update(value)
        self._env.cal = candidate
        self._record_update('set_readout_drive_controls', updates)
        return self.status()

    def _admit_experiment(self):
        from .experiment_budget import BudgetExhausted
        if self._max_experiments is not None and self.experiment_count >= self._max_experiments:
            raise BudgetExhausted('Standard experiment budget exhausted')

    def run_scan(self, kind, requests, shots=1024):
        from .flux_scan import run_scan
        return run_scan(self, kind, requests, shots)

    def acquire(self, blocks, *, preparation=('0', '0'), basis=('Z', 'Z'), shots=1024, analysis_phases=(0.0, 0.0)):
        if not self._in_scan:
            self._admit_experiment()
        if type(shots) is not int or shots <= 0:
            raise ValueError('shots must be a positive integer')
        used = self._env.ledger
        if self._max_acquisitions is not None and used.experiments >= self._max_acquisitions or (self._max_shots is not None and used.shots + shots > self._max_shots):
            raise ValueError('Acquisition budget exceeded')
        seq = prepare_product(self._env, list(preparation))
        analysis_offset = np.asarray(analysis_phases, float).copy()
        if analysis_offset.shape != (2,) or not np.isfinite(analysis_offset).all():
            raise ValueError('Invalid analysis phases')
        frames = np.zeros(2)
        for block in blocks:
            if isinstance(block, dict):
                if set(block) == {'gate', 'repetitions'} and block['gate'] == 'cz':
                    gate, advance = compile_gate(self._env, repetitions=block['repetitions'])
                    seq.extend(gate)
                    frames += advance
                elif set(block) == {'xy'}:
                    pulses = []
                    for command in block['xy']:
                        allowed = {'qubit', 'angle', 'phase', 'amp', 'duration', 'drag', 'freq'}
                        if set(command) - allowed or command.get('qubit') not in (0, 1):
                            raise ValueError('Invalid XY command')
                        args = {k: v for k, v in command.items() if k != 'qubit'}
                        from .flux_coupler import finite
                        for name, value in args.items():
                            limits = {'angle': (-100.0, 100.0), 'phase': (-100.0, 100.0), 'amp': (-2.0, 2.0), 'duration': (1.0, 2000.0), 'drag': (-5.0, 5.0), 'freq': (0.1, 15.0)}
                            finite(value, name, *limits[name])
                        args['phase'] = args.get('phase', 0.0) + frames[command['qubit']]
                        pulses.append(self._env.x_pulse(command['qubit'], **args))
                    if not pulses:
                        raise ValueError('Empty XY block')
                    seq.append(Segment(pulses))
                elif set(block) == {'duration', 'pulses'}:
                    seq.append(FluxSegment(tuple((FluxPulse(**p) for p in block['pulses'])), block['duration']))
                else:
                    raise ValueError('Unknown sequence block field')
            else:
                from .flux_coupler import finite
                finite(block, 'idle', 0.0, 1000000.0)
                seq.append(float(block))
        analysis(self._env, seq, list(basis), frames + analysis_offset)
        self._env.sync_physics()
        controls = {k: getattr(self._env.cal, k) for k in ('flux_bias', 'coupler_flux_bias', 'flux_mix', 'flux_inverse_tau', 'flux_inverse_tail')}
        key = repr((self._env.dev.to_json(), controls, seq, self._env.sim.opts))
        hit = key in self._observation_cache
        if not hit:
            pops = self._env.sim.run(seq)
            duration = self._env.sim._schedule(seq)[0]
            if len(self._observation_cache) >= 256:
                self._observation_cache.pop(next(iter(self._observation_cache)))
            self._observation_cache[key] = (pops, duration)
        pops, duration = self._observation_cache[key]
        iq = self._env._sample_simulation(pops, duration, shots, None, 'flux_sequence')
        if not self._in_scan:
            self.experiment_count += 1
        if self._directory:
            index = used.experiments
            np.savez_compressed(self._directory / f'{index:04d}_iq.npz', **{f'q{k}': v for k, v in iq.items()})
            (self._directory / f'{index:04d}.json').write_text(json.dumps(dict(blocks=blocks, preparation=preparation, basis=basis, shots=shots, analysis_phases=list(analysis_phases), dynamics_cache_hit=hit, status=self.status()), indent=2))
        return iq
