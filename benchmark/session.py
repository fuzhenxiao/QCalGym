from dataclasses import asdict
from backend.session import Session
from backend.flux_benchmark_session import FluxBenchmarkSession, control_schema
from backend.benchmark.controls import control_catalog
from backend.benchmark.session import Budget, serial
from backend.probes import install


def differences(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in set(a) | set(b) for p in differences(a.get(k, {}), b.get(k, {}), path + (k,))]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, path + (i,))]
    return [] if a == b else [path]


class TuneMixin:
    def _update(self, updates):
        before = self.env.cal.copy()
        try:
            super()._update(updates)
            allowed = {(c['field'], c['index'], *c['path']) for c in self.task.changes}
            if any(p not in allowed for p in differences(serial(asdict(before)), serial(asdict(self.env.cal)))):
                raise ValueError('Only the named tuning parameters may change')
        except Exception:
            self.env.cal = before
            self.env.sync_physics()
            raise


class TuneSession(TuneMixin, Session):
    pass


class TuneFluxSession(TuneMixin, FluxBenchmarkSession):
    pass


def create_session(directory, task, env, initial, seed=0, experiments=None, shots=None):
    tune = task.track == 'tune'
    flux = task.backend == 'flux'
    cls = (TuneFluxSession if flux else TuneSession) if tune else (FluxBenchmarkSession if flux else Session)
    limit = experiments if experiments is not None else (None if tune else 100)
    shot_limit = shots if shots is not None else (7_000_000 if tune else None)
    session = cls(directory, task, env, initial, seed=seed, experiment_limit=limit,
                  budget=Budget(None, shot_limit, None),
                  experiment_catalog={} if flux else install(env.dev.n, env.dev.topology))
    session.objective['task_id'] = task.id
    session.objective['track'] = task.track
    if tune:
        fields = (control_schema(env.dev) if flux else control_catalog(env.dev.n, env.dev.edges, calibration=env.cal, device=env.dev))['fields']
        space = []
        state = serial(asdict(env.cal))
        for c in task.changes:
            schema = fields[c['field']]['value']
            start = state[c['field']][c['index']]
            for key in c['path']:
                schema = schema['items'] if schema['type'] == 'array' else schema['properties'][key]
                start = start[key]
            space.append(dict(field=c['field'], index=c['index'], path=c['path'], start=start,
                              minimum=schema['minimum'], maximum=schema['maximum']))
        session.objective['parameters_to_fix'] = space
    return session
