import numpy as np
from .core import Segment
from .gates import compiled, installed_idle
from .experiments.common import Result, new_fig
from .benchmark.catalog import register_experiment, catalog

def machine_probe(env, preparation, basis, gate='x90', k=0, edge=None, phases=None, gates=None, duration=1000.0, repetitions=1, shots=1024):
    n = env.dev.n
    if len(preparation) != n or len(basis) != n:
        raise ValueError('One preparation and basis per qubit required')
    seq = []
    for q, state in enumerate(preparation):
        if state == '1':
            seq.append(Segment([env.x_pulse(q)]))
        elif state == '+':
            seq.append(Segment([env.y_pulse(q, -np.pi / 2)]))
        elif state == '+i':
            seq.append(Segment([env.x_pulse(q, -np.pi / 2)]))
    frames = np.zeros(n)
    for _ in range(repetitions):
        if gate in {'cnot', 'cz', 'zz_idle'}:
            if edge is None or tuple(edge) not in env.dev.edges:
                raise ValueError('An installed directed edge is required')
            block, advance = compiled(env, gate, tuple(edge), frames=frames)
            seq.extend(block)
            frames += advance
        elif gate == 'id':
            block, advance = installed_idle(env, duration, frames=frames)
            seq.extend(block)
            frames += advance
        elif gate == 'parallel':
            if phases is None or gates is None or len(phases) != n or (len(gates) != n):
                raise ValueError('parallel requires one phase and gate per qubit')
            seq.append(Segment([env.x_pulse(q, np.pi / 2 if g == 'x90' else np.pi, phase=phases[q] + frames[q]) for q, g in enumerate(gates)]))
        else:
            seq.append(Segment([env.x_pulse(k, np.pi / 2 if gate == 'x90' else np.pi, phase=frames[k])]))
    for q, axis in enumerate(basis):
        if axis == 'X':
            seq.append(Segment([env.x_pulse(q, np.pi / 2, phase=np.pi / 2 + frames[q])]))
        elif axis == 'Y':
            seq.append(Segment([env.x_pulse(q, np.pi / 2, phase=frames[q])]))
    iq = env.measure(seq, shots, list(range(n)), name='machine_probe')
    bits = np.array([env.assign(iq[q], q) for q in range(n)])
    indices = np.ravel_multi_index(bits, (2,) * n)
    counts = np.bincount(indices, minlength=2 ** n)
    fig, ax = new_fig('Installed machine probe', 'joint outcome', 'probability')
    ax.bar(range(2 ** n), counts / shots)
    ax.set_xticks(range(2 ** n), [format(i, f'0{n}b') for i in range(2 ** n)])
    return Result('machine_probe', {'counts': counts, 'probabilities': counts / shots}, {'ok': True, 'params': {}, 'model': 'measured_histogram_not_fidelity'}, {'shots': shots, 'gate': gate, 'includes_spam': True}, fig)

def install(n, topology=None, include_hidden=False):

    def array(items, length=n):
        return {'type': 'array', 'minItems': length, 'maxItems': length, 'items': items}
    props = {'preparation': array({'type': 'string', 'enum': ['0', '1', '+', '+i']}), 'basis': array({'type': 'string', 'enum': ['X', 'Y', 'Z']}), 'gate': {'type': 'string', 'enum': ['x90', 'x180', 'id', 'parallel', 'cnot', 'cz', 'zz_idle'], 'default': 'x90'}, 'k': {'type': 'integer', 'minimum': 0, 'maximum': n - 1, 'default': 0}, 'edge': dict(array({'type': 'integer', 'minimum': 0, 'maximum': n - 1}, 2), type=['array', 'null'], default=None), 'phases': dict(array({'type': 'number', 'minimum': -20, 'maximum': 20}), type=['array', 'null'], default=None), 'gates': dict(array({'type': 'string', 'enum': ['x90', 'x180']}), type=['array', 'null'], default=None), 'duration': {'type': 'number', 'minimum': 1, 'maximum': 20000, 'default': 1000.0}, 'repetitions': {'type': 'integer', 'minimum': 1, 'maximum': 32, 'default': 1}, 'shots': {'type': 'integer', 'minimum': 16, 'maximum': 4096, 'default': 1024}}
    register_experiment('machine_probe', machine_probe, {'description': 'Prepare product Pauli states, execute installed local/parallel/edge/idle gates, measure joint outcomes in selected bases. Includes SPAM; no author score returned.', 'parameters': {'type': 'object', 'required': ['preparation', 'basis'], 'additionalProperties': False, 'properties': props}}, lambda p: 1)
    from .diagnostic_scans import install as diagnostic_scans
    diagnostic_scans(n)
    from .standard_scans import install as standard_scans
    standard_scans(n)
    from .new12_scans import install as new12_scans
    new12_scans(n)
    from .workflow_experiments import install as workflow_experiments
    from .reproduced_experiments import install as reproduced_experiments
    from .qubic_experiments import install as qubic_experiments
    workflow_experiments(n)
    reproduced_experiments(n)
    qubic_experiments(n)
    return catalog(n, topology=topology, include_hidden=include_hidden)
