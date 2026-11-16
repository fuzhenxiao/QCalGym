import numpy as np
import matplotlib.pyplot as plt
from .core import Segment
from .gates import compiled
from .hardware_probes import prepare_product, drive_response
from .probes import machine_probe
from .experiments.common import Result
from .acquisition_cache import reuse_acquisitions

def linear(x, y):
    a, b = np.polyfit(x, y, 1)
    return dict(slope=float(a), intercept=float(b), residual_rms=float(np.sqrt(np.mean((y - a * np.asarray(x) - b) ** 2))))

@reuse_acquisitions
def gate_phase_scan(env, gate, n_list, shots=4096):
    ns = np.asarray(n_list)
    xy = {}
    phases = {}
    contrast = {}
    fits = {}
    branches = ['q0', 'q1'] if gate == 'cnot' else ['neighbour0', 'neighbour1']
    values = {b: [] for b in branches}
    for n in ns:
        if gate == 'cnot':
            quadratures = []
            for axis in ['X', 'Y']:
                r = machine_probe(env, ['+', '+'], [axis, axis], gate=gate, edge=[0, 1], repetitions=int(n), shots=shots)
                plt.close(r.figure)
                p = np.asarray(r.points['probabilities']).reshape(2, 2)
                quadratures.append([1 - 2 * p[1, :].sum(), 1 - 2 * p[:, 1].sum()])
            for k, b in enumerate(branches):
                values[b].append([quadratures[0][k], quadratures[1][k]])
        else:
            for k, b in enumerate(branches):
                q = []
                for axis in ['X', 'Y']:
                    r = machine_probe(env, ['+', str(k)], [axis, 'Z'], gate=gate, edge=[0, 1], repetitions=int(n), shots=shots)
                    plt.close(r.figure)
                    p = np.asarray(r.points['probabilities']).reshape(2, 2)
                    q.append(1 - 2 * p[1, :].sum())
                values[b].append(q)
    for b in branches:
        xy[b] = np.asarray(values[b])
        phases[b] = np.arctan2(xy[b][:, 1], xy[b][:, 0])
        contrast[b] = np.linalg.norm(xy[b], axis=1)
    if gate == 'cnot':
        series = {b: np.unwrap(phases[b]) for b in branches}
    else:
        series = {'conditional_residual': np.unwrap(np.angle(np.exp(1j * (phases['neighbour1'] - phases['neighbour0'] - np.pi * ns))))}
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), layout='constrained', sharex=True)
    for b, y in series.items():
        fits[b] = linear(ns, y)
        a = fits[b]['slope']
        intercept = fits[b]['intercept']
        axes[0].plot(ns, y, 'o', label=f'{b}: slope={a:.4f} rad/gate')
        axes[0].plot(ns, a * ns + intercept)
    for b, y in contrast.items():
        axes[1].plot(ns, y, 'o-', label=b)
    axes[0].set(ylabel='Phase residual (rad)', title=f'{gate.upper()} repeated-gate phase')
    axes[1].set(xlabel='Installed gate repetitions', ylabel='Transverse contrast', ylim=(0, 1.1))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    return Result('gate_phase_scan', dict(n=ns, XY=xy, phase_rad=phases, contrast=contrast, series=series), dict(status='finished', model='phase_slopes', params=fits), dict(shots=shots, gate=gate, includes_spam=True), fig)

@reuse_acquisitions
def gate_fringe(env, angles, shots=4096):
    xs = np.asarray(angles)
    ys = {}
    fits = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    basis = np.array([np.ones(len(xs)), np.cos(xs), np.sin(xs)]).T
    for b in [0, 1]:
        y = []
        for phi in xs:
            seq = prepare_product(env, ['+', str(b)])
            block, frames = compiled(env, 'cz', (0, 1), frames=np.zeros(2))
            seq.extend(block)
            seq.append(Segment([env.x_pulse(0, np.pi / 2, phase=np.pi / 2 + frames[0] + phi)]))
            iq = env.measure(seq, shots, [0], name='cz_analysis_fringe')[0]
            y.append(env.p1(iq, 0))
        ys[str(b)] = np.asarray(y)
        coef = np.linalg.lstsq(basis, y, rcond=None)[0]
        fits[str(b)] = dict(phase=float(np.arctan2(-coef[2], coef[1])), contrast=float(2 * np.hypot(coef[1], coef[2])))
        ax.plot(xs, y, 'o', ms=3, label=f'Neighbour |{b}>')
        ax.plot(xs, basis @ coef)
    separation = float(np.angle(np.exp(1j * (fits['1']['phase'] - fits['0']['phase']))))
    ax.set(xlabel='Analysis pulse phase (rad)', ylabel='q0 P(1)', title=f'Installed CZ fringes: separation={separation:.4f} rad', ylim=(-0.05, 1.05))
    ax.grid(alpha=0.25)
    ax.legend()
    return Result('gate_fringe', dict(angle=xs, p1=ys), dict(status='finished', model='sinusoidal_fringe', params=dict(branches=fits, separation_rad=separation)), dict(shots=shots), fig)

def drive_phase_scan(env, source, freq, amp, duration, phases, shots=4096):
    vals = []
    target = 1 - source
    for phase in phases:
        row = []
        for axis in ['X', 'Y']:
            basis = ['Z', 'Z']
            basis[target] = axis
            r = drive_response(env, source, [freq], amp, duration, ['0', '0'], basis, phase=phase, shots=shots)
            plt.close(r.figure)
            row.append(1 - 2 * r.points['p1'][0, target])
        vals.append(row)
    vals = np.asarray(vals)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    for k, a in enumerate(['X', 'Y']):
        ax.plot(phases, vals[:, k], 'o-', label=f'q{target} {a}')
    ax.set(xlabel='Source phase (rad)', ylabel='Neighbour Pauli expectation', ylim=(-1.1, 1.1), title='Source-path phase response')
    ax.grid(alpha=0.25)
    ax.legend()
    return Result('drive_phase_scan', dict(phase=phases, XY=vals), dict(status='finished', params=dict(max_transverse=float(np.linalg.norm(vals, axis=1).max()))), dict(shots=shots), fig)

@reuse_acquisitions
def readout_window_scan(env, k, durations, shots=4096):
    before = env.cal.readout_duration[k]
    errors = []
    try:
        for t in durations:
            env.cal.readout_duration[k] = float(t)
            row = []
            for b in [0, 1]:
                prep = ['0'] * env.dev.n
                prep[k] = str(b)
                iq = env.measure(prepare_product(env, prep), shots, [k], name='readout_window_scan')[k]
                p = env.p1(iq, k)
                row.append(p if b == 0 else 1 - p)
            errors.append(row)
    finally:
        env.cal.readout_duration[k] = before
    errors = np.asarray(errors)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    for j, label in enumerate(['Prepared 0 -> assigned 1', 'Prepared 1 -> assigned 0']):
        ax.plot(np.asarray(durations) / 1000, errors[:, j], 'o-', label=label)
    ax.set(xlabel='Integration duration (us)', ylabel='Assignment error', title='Readout-window scan, fixed classifier')
    ax.grid(alpha=0.25)
    ax.legend()
    return Result('readout_window_scan', dict(duration_ns=durations, errors=errors), dict(status='finished', params=dict(best_duration_ns=float(durations[int(np.argmin(errors.mean(axis=1)))]))), dict(shots=shots, controls_restored=True), fig)

def install(n):
    from .benchmark.catalog import register_experiment
    num = lambda a, b: dict(type='number', minimum=a, maximum=b)
    arr = lambda item: dict(type='array', items=item, minItems=3, maxItems=41)
    shots = dict(type='integer', minimum=16, maximum=4096, default=4096)
    specs = {}

    def add(name, fn, props, required, count, desc):
        spec = dict(description=desc, parameters=dict(type='object', properties={**props, 'shots': shots}, required=required, additionalProperties=False))
        register_experiment(name, fn, spec, count)
        specs[name] = spec
    add('readout_window_scan', readout_window_scan, dict(k=dict(type='integer', minimum=0, maximum=n - 1), durations=arr(num(100, 20000))), ['k', 'durations'], lambda p: 2 * len(p['durations']), 'Two prepared states versus integration time, fixed classifier; restores integration duration.')
    if n == 2:
        add('gate_phase_scan', gate_phase_scan, dict(gate=dict(type='string', enum=['cnot', 'cz']), n_list=arr(dict(type='integer', minimum=0, maximum=16))), ['gate', 'n_list'], lambda p: (2 if p['gate'] == 'cnot' else 4) * len(p['n_list']), 'Repeated installed-gate X/Y scan; CNOT ++ phases or CZ conditional phase residual relative to N*pi, and measured contrast.')
        add('gate_fringe', gate_fringe, dict(angles=arr(num(-20, 20))), ['angles'], lambda p: 2 * len(p['angles']), 'Installed CZ analysis-phase scan under both neighbour states.')
        add('drive_phase_scan', drive_phase_scan, dict(source=dict(type='integer', minimum=0, maximum=1), freq=num(0.1, 15), amp=num(-2, 2), duration=num(1, 20000), phases=arr(num(-20, 20))), ['source', 'freq', 'amp', 'duration', 'phases'], lambda p: 2 * len(p['phases']), 'Source phase scan measuring neighbour X/Y at a user-specified carrier.')
    from .reference_scans import install as install_reference_scans
    specs.update(install_reference_scans(n))
    from .quadrature_scan import install as install_quadrature_scan
    specs.update(install_quadrature_scan(n))
    return specs
