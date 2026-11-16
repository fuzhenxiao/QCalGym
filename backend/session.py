from dataclasses import asdict
import json
from .benchmark.session import serial
from .benchmark.session import Session as AcquisitionSession
from .performance import evaluate
from .benchmark.catalog import acquisition_count, prepare
INSTRUCTION = 'Calibrate the device so that all applicable metrics in the current Machine Performance pass. Success is determined solely by the final physical performance; submission is optional. initial_machine_performance is a fixed snapshot at calibration revision 0, not live feedback. Control updates change the calibration and its revision, but do not refresh this snapshot. Validate updated settings with public experiments; formal scoring runs only after the episode ends. Interact with the device only through experiments, calibration parameter updates, and submission. Units are GHz, ns, and rad. Use English for all written analysis, prose in tool arguments, visual-inspection requests, and final reports.'

def public_objective(initial, family):
    return {'instruction': INSTRUCTION, 'initial_machine_performance': initial, 'performance_snapshot': {'kind': 'initial', 'calibration_revision': 0, 'live': False, 'formal_scoring': 'after_episode'}, 'gate_family': family}

class Session(AcquisitionSession):

    def save_checkpoint(self):
        state = {'receipt_count': len(self.receipts), 'rng': self.env.ro.rng.bit_generator.state, 'ledger': asdict(self.env.ledger), 'experiments': self.experiment_count}
        path = self.directory / 'checkpoint.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(serial(state), allow_nan=False))
        tmp.replace(path)

    def dispatch(self, action, params=None, request_id=None):
        with self.lock:
            result = super().dispatch(action, params, request_id)
            if action in ('run_experiment', 'update_calibration', 'submit'):
                self.save_checkpoint()
            return result

    def __init__(self, directory, task, env, initial, experiment_limit=100, extra_updates=(), **kwargs):
        self.task = task
        self.experiment_limit = experiment_limit
        self.experiment_count = 0
        self.initial_performance = initial
        allowed = set(asdict(env.cal)) - {'drive_line', 'cnot_pre_phase'} | set(extra_updates) & {'cnot_pre_phase'}
        objective = public_objective(initial, task.family)
        from .parametric import enabled, public_apparatus
        if enabled(env.dev):
            objective['apparatus'] = public_apparatus()
        super().__init__(directory, env=env, objective=objective, allowed_updates=allowed, **kwargs)

    def _dispatch(self, action, params, request_id):
        if action == 'run_experiment' and request_id not in self.receipts and (self.experiment_limit is not None) and (self.experiment_count >= self.experiment_limit):
            return {'status': 'failed', 'error_type': 'budget_exhausted', 'error': 'Experiment budget exhausted; updates and submit remain available'}
        if action == 'run_experiment' and params.get('name') in self.experiment_catalog:
            kwargs = prepare(params['name'], params.get('parameters', {}), self.env)
            if acquisition_count(params['name'], kwargs) > 441:
                raise ValueError('At most 441 acquisitions per experiment')
        before = self.env.ledger.shots
        result = super()._dispatch(action, params, request_id)
        if action == 'run_experiment' and self.env.ledger.shots > before:
            self.experiment_count += 1
        return result

    def status(self):
        value = super().status()
        value['used']['experiment_calls'] = self.experiment_count
        value['budget']['experiment_calls'] = self.experiment_limit
        return value

    def final_performance(self, progress=lambda row: None):
        return evaluate(self.env, self.task.family, progress=progress)
