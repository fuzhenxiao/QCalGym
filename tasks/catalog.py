import ast
import json
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENES = ROOT.parent / 'scenes'


@dataclass(frozen=True)
class Task:
    id: str
    scene: str
    tier: str
    cold: bool
    changes: tuple
    track: str

    @property
    def definition(self):
        return json.loads((SCENES / f'{self.scene}.json').read_text())

    @property
    def family(self):
        return self.definition['family']

    @property
    def backend(self):
        return self.definition['backend']


def load_tasks(track=None):
    tracks = (track,) if track else ('troubleshoot', 'tune')
    return tuple(Task(**dict(row, changes=tuple(row['changes'])), track=name)
                 for name in tracks for row in json.loads((ROOT / f'{name}.json').read_text()))


def get_task(task_id):
    return next(task for task in load_tasks() if task.id == task_id)


def apply_changes(snapshot, changes):
    snapshot = deepcopy(snapshot)
    for change in changes:
        field, index, path = (change[k] for k in ('field', 'index', 'path'))
        if path:
            target = snapshot[field][index]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = change['after']
        else:
            snapshot.setdefault(field, {})[index] = change['after']
    return snapshot


def install_snapshot(env, snapshot):
    if 'flux_coupler' in env.dev.physics:
        from backend.coupler_session import FluxCouplerSession
        session = FluxCouplerSession(env.dev, cal=env.cal, solver_options=env.sim.opts, solver_mode=env.sim.mode)
        values = {k: {int(i): v for i, v in row.items()} for k, row in snapshot.items() if row}
        names = {'flux_bias': 'qubit_bias', 'coupler_flux_bias': 'coupler_bias', 'flux_mix': 'mix',
                 'flux_inverse_tau': 'inverse_tau', 'flux_inverse_tail': 'inverse_tail'}
        session.set_readout_drive_controls(**{k: v for k, v in values.items() if k not in names and k != 'native_flux_gate'})
        session.set_flux_controls(**{names[k]: v[0] if k in ('coupler_flux_bias', 'flux_mix') else v
                                     for k, v in values.items() if k in names})
        if 'native_flux_gate' in values:
            session.set_native_gate(**values['native_flux_gate'][0])
        env.cal = session._env.cal.copy()
    else:
        from backend.benchmark.session import Session, Budget
        updates = []
        for field, entries in snapshot.items():
            for index, value in entries.items():
                parsed = ast.literal_eval(index)
                updates.append(dict(field=field, index=list(parsed) if isinstance(parsed, tuple) else parsed, value=value))
        with tempfile.TemporaryDirectory() as directory:
            session = Session(Path(directory) / 'host', env=env, budget=Budget(None, None, None), experiment_runtime=False)
            for start in range(0, len(updates), 64):
                session._update(updates[start:start + 64])
            env.cal = session.env.cal.copy()
    env.sync_physics()
    return env


def build(task, seed=0):
    from backend.device import DeviceConfig, QubitParams, CouplingParams, ResonatorParams, CalibrationStore
    from backend.flux_coupler import FluxCouplerCalibration
    from backend.env import Env
    scene = task.definition
    data = deepcopy(scene['device'])
    data['qubits'] = [QubitParams(**q) for q in data['qubits']]
    data['couplings'] = [CouplingParams(**c) for c in data['couplings']]
    data['resonators'] = [ResonatorParams(**r) for r in data['resonators']]
    if data['drive_crosstalk'] is not None:
        data['drive_crosstalk'] = [[complex(*z) for z in row] for row in data['drive_crosstalk']]
    cal = FluxCouplerCalibration() if task.backend == 'flux' else CalibrationStore()
    for field, values in scene['initial_controls'].items():
        setattr(cal, field, {int(k): v for k, v in values.items()})
    options = None if task.backend == 'flux' else {'exact_flat': True, 'exact_idle': True, 'method': 'dop853'}
    env = Env(DeviceConfig(**data), cal, seed=seed, solver_mode='qutip', solver_options=options)
    if not task.cold:
        install_snapshot(env, apply_changes(scene['reference_controls'], task.changes))
    return env
