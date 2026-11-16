import numpy as np
import matplotlib.pyplot as plt
from .experiments.common import Result
from .probes import machine_probe

def measured(env, prep, basis, gate, shots, **kw):
    r = machine_probe(env, prep, basis, gate=gate, shots=shots, **kw)
    plt.close(r.figure)
    p = np.asarray(r.points['probabilities'])
    m = [float(p @ (1 - 2 * (np.arange(4) >> 1 - k & 1))) for k in range(2)]
    return (p, m)

def cnot_amplitude_scan(env, amps, shots=2048):
    old = env.cal.cr_amp[0, 1]
    values = []
    try:
        for amp in amps:
            env.cal.cr_amp[0, 1] = amp
            p, _ = measured(env, ['0', '0'], ['Z', 'Z'], 'cnot', shots, edge=[0, 1])
            q, _ = measured(env, ['+', '0'], ['X', 'X'], 'cnot', shots, edge=[0, 1])
            z, _ = measured(env, ['+', '+'], ['X', 'X'], 'cnot', shots, edge=[0, 1])
            values.append([p[0], q[0] + q[3], z[0]])
    finally:
        env.cal.cr_amp[0, 1] = old
    values = np.asarray(values)
    cost = 1 - values.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    for j, label in enumerate(['00 input: P(00)', '+0 input: P(XX same)', '++ input: P(++)']):
        ax.plot(amps, values[:, j], 'o-', label=label)
    ax.plot(amps, cost, 'o--', label='Mean deviation')
    ax.set(xlabel='CR amplitude', ylabel='Measured probability / deviation', title='Installed CNOT amplitude response')
    ax.grid(alpha=0.25)
    ax.legend()
    return Result('cnot_amplitude_scan', dict(amp=amps, probabilities=values, cost=cost), dict(status='finished', params=dict(best_amplitude=float(amps[int(np.argmin(cost))]))), dict(shots=shots, controls_restored=True), fig)

def gate_quadratures(env, gate, qubits, shots=4096):
    xy = {}
    phases = {}
    if gate == 'cnot':
        xy = {str(k): [] for k in qubits}
        for axis in ['X', 'Y']:
            _, m = measured(env, ['+', '+'], [axis, axis], gate, shots, edge=[0, 1])
            for k in qubits:
                xy[str(k)].append(m[k])
        phases = {k: float(np.arctan2(v[1], v[0])) for k, v in xy.items()}
    else:
        for k in qubits:
            for b in [0, 1]:
                prep = [str(b)] * 2
                prep[k] = '+'
                v = []
                for axis in ['X', 'Y']:
                    basis = ['Z', 'Z']
                    basis[k] = axis
                    _, m = measured(env, prep, basis, gate, shots, edge=[0, 1])
                    v.append(m[k])
                key = f'{k}:{b}'
                xy[key] = v
                phases[key] = float(np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.pi * b))))
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    t = np.linspace(0, 2 * np.pi, 101)
    ax.plot(np.cos(t), np.sin(t), color='lightgray')
    for k, v in xy.items():
        ax.plot([0, v[0]], [0, v[1]], 'o-', label=k)
    ax.set(xlabel='X', ylabel='Y', title=f'{gate.upper()} measured output quadratures', xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend()
    return Result('gate_quadratures', dict(XY=xy), dict(status='finished', params=dict(phases=phases)), dict(shots=shots), fig)

def idle_coherence_scan(env, delays, shots=512):
    rows = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for k in [0, 1]:
        for b in [0, 1]:
            v = []
            for duration in delays:
                prep = [str(b)] * 2
                prep[k] = '+'
                row = []
                for axis in ['X', 'Y']:
                    basis = ['Z', 'Z']
                    basis[k] = axis
                    _, m = measured(env, prep, basis, 'id', shots, duration=duration)
                    row.append(m[k])
                v.append(row)
            v = np.asarray(v)
            rows[f'{k}:{b}'] = v
            axes[k].plot(delays, v[:, 0], 'o-', label=f'neighbour {b}: X')
            axes[k].plot(delays, v[:, 1], 'o--', label=f'neighbour {b}: Y')
        axes[k].set(xlabel='Wait (ns)', ylabel='Pauli expectation', title=f'q{k}: installed echo idle', ylim=(-1.1, 1.1))
        axes[k].grid(alpha=0.25)
        axes[k].legend(fontsize=8)
    return Result('idle_coherence_scan', dict(delay_ns=delays, XY=rows), dict(status='not_fitted', params={}), dict(shots=shots), fig)

def install(n):
    if n != 2:
        return {}
    from .benchmark.catalog import register_experiment
    specs = {}
    shots = dict(type='integer', minimum=16, maximum=4096, default=1024)

    def add(name, fn, props, count, description):
        spec = dict(description=description, parameters=dict(type='object', properties={**props, 'shots': shots}, required=list(props), additionalProperties=False))
        register_experiment(name, fn, spec, count)
        specs[name] = spec
    array = lambda lo, hi: dict(type='array', items=dict(type='number', minimum=lo, maximum=hi), minItems=3, maxItems=41)
    add('cnot_amplitude_scan', cnot_amplitude_scan, dict(amps=array(-2, 2)), lambda p: 3 * len(p['amps']), 'Installed CNOT amplitude response for 00/ZZ, +0/XX and ++/XX. Restores amplitude; three acquisitions per point.')
    add('gate_quadratures', gate_quadratures, dict(gate=dict(type='string', enum=['cnot', 'cz']), qubits=dict(type='array', items=dict(type='integer', minimum=0, maximum=1), minItems=1, maxItems=2, uniqueItems=True)), lambda p: 2 if p['gate'] == 'cnot' else 4 * len(p['qubits']), 'Measured X/Y output vectors. CNOT ++ samples both qubits together; CZ uses both neighbour states.')
    add('idle_coherence_scan', idle_coherence_scan, dict(delays=array(200, 20000)), lambda p: 8 * len(p['delays']), 'Installed idle X/Y versus waiting time for each qubit and both neighbour states; eight acquisitions per point.')
    return specs
