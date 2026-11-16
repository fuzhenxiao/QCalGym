import itertools
from dataclasses import asdict
from collections import OrderedDict
from copy import deepcopy
import os
import json
from pathlib import Path
import numpy as np
import qutip as qt
from scipy.special import ndtr
from .core import Segment
from .gates import rotation, product, compiled, two_target, installed_idle

def average_channel(evolve, target):
    d = len(target)
    eye = np.eye(d, dtype=complex)
    diagonal = [evolve(np.outer(v, v.conj())) for v in eye]
    trace = sum((np.trace(x).real for x in diagonal))
    overlap = sum((np.vdot(target[:, i], diagonal[i] @ target[:, i]) for i in range(d)))
    for i in range(d):
        for j in range(i + 1, d):
            p = (eye[i] + eye[j]) / np.sqrt(2)
            q = (eye[i] + 1j * eye[j]) / np.sqrt(2)
            ep, eq = (evolve(np.outer(p, p.conj())), evolve(np.outer(q, q.conj())))
            eij = (2 * ep - diagonal[i] - diagonal[j] + 1j * (2 * eq - diagonal[i] - diagonal[j])) / 2
            overlap += np.vdot(target[:, i], eij @ target[:, j]) + np.vdot(target[:, j], eij.conj().T @ target[:, i])
    return {'fidelity': float((overlap.real + trace) / (d * (d + 1))), 'leakage': float(max(0.0, 1 - trace / d)), 'input_states': d * d}

def _freeze(value):
    if isinstance(value, dict):
        return tuple(((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple((_freeze(v) for v in value))
    return value

def channel_score(env, sequence, target, phases=None, *, method='auto'):
    env.sync_physics()
    if 'flux_coupler' in env.dev.physics:
        if method not in ('auto', 'reference'):
            raise ValueError('Coupler channels require the full physical channel route')
        from .coupler_runtime import channel_score as coupler_score
        return coupler_score(env, sequence, target, phases)
    sim = env.sim
    if method not in ('auto', 'reference', 'operators', 'certified', 'jumps'):
        raise ValueError('Unknown channel method')
    selected = ('operators' if sim.mode == 'qutip' and env.dev.n <= 2 else 'reference') if method == 'auto' else method
    if method == 'auto' and getattr(sim, 'certified_long_operations', False) and (env.dev.n >= 3):
        selected = 'jumps'
    if method == 'auto' and getattr(sim, 'certified_short_layers', False) and (env.dev.n >= 3) and (len(sequence) == 1) and isinstance(sequence[0], Segment) and (0 < sequence[0].duration <= 50):
        selected = 'certified'
    if method == 'auto' and selected == 'jumps':
        from .three_qubit_runtime import supported
        if supported(env.dev):
            selected = 'three_jumps'
    from .parametric import enabled
    if method == 'auto' and selected == 'three_jumps' and enabled(env.dev):
        total = sum((x.duration if isinstance(x, Segment) else float(x) for x in sequence))
        if 0 < total <= 600:
            selected = 'three_short'
    if method == 'auto' and enabled(env.dev) and all((not isinstance(x, Segment) for x in sequence)):
        selected = 'three_idle'
    if selected in ('operators', 'certified', 'jumps', 'three_jumps') and sim.mode != 'qutip':
        raise ValueError('Bounded and operator channels require full Lindblad mode')
    from .hardware import dynamics_enabled
    if dynamics_enabled(env.dev):
        if method not in ('auto', 'reference'):
            raise ValueError('Optional hardware uses the full-sequence reference channel')
        selected = 'extended_batch' if method == 'auto' and env.dev.n <= 2 else 'reference'
    target = np.asarray(target, dtype=complex)
    key = (_freeze(asdict(env.dev)), _freeze(getattr(sim, 'hardware_controls', {})), sim.noise_order, sim.noise_max_order, sim.noise_tolerance, sim.mode, _freeze(sim.opts), sim.exact_flat, sim.exact_idle, selected, os.environ.get('QCAL_SIM_BACKEND', 'off'), tuple((_freeze(asdict(x)) if isinstance(x, Segment) else float(x) for x in sequence)), tuple((env.cal.drive_freq[k] for k in range(env.dev.n))), target.shape, target.tobytes(), tuple(np.zeros(env.dev.n) if phases is None else phases))
    if not hasattr(env, '_channel_score_cache'):
        env._channel_score_cache = OrderedDict()
    cache = env._channel_score_cache
    if key in cache:
        cache.move_to_end(key)
        return {**deepcopy(cache[key]), 'cache_hit': True}
    disk_hit = False
    if getattr(sim, 'channel_cache_path', None):
        from .channel_cache import get_or_compute
        from .three_qubit_runtime import supported
        result, disk_hit = get_or_compute(sim.channel_cache_path, key, lambda: _channel_score_reference(env, sequence, target, phases, method=selected), per_key_lock=supported(env.dev))
    else:
        result = _channel_score_reference(env, sequence, target, phases, method=selected)
    cache[key] = deepcopy(result)
    if len(cache) > 2048:
        cache.popitem(last=False)
    return {**result, 'cache_hit': disk_hit}

def _channel_score_reference(env, sequence, target, phases=None, *, method='reference'):
    vectors, labels = env.sim.dressed()
    vectors = vectors.astype(complex).copy()
    for j, label in enumerate(labels):
        bare = np.ravel_multi_index(label, env.sim.dims)
        vectors[:, j] *= np.exp(-1j * np.angle(vectors[bare, j]))
    codes = np.array(list(itertools.product((0, 1), repeat=env.dev.n)))
    basis = vectors[:, [labels.index(tuple(code)) for code in codes]]
    phases = np.zeros(env.dev.n) if phases is None else np.asarray(phases)
    duration = env.sim.sequence_duration(sequence)
    frame = np.exp(1j * (2 * np.pi * duration * codes @ np.array([env.cal.drive_freq[k] for k in range(env.dev.n)]) + codes @ phases))
    if method == 'operators':
        from .operator_channel import score
        return {**score(env, sequence, target, basis, frame), 'duration_ns': float(duration)}
    if method == 'certified':
        from .certified_channel import score
        return {**score(env.sim, sequence, target, basis, frame), 'duration_ns': float(duration)}
    if method == 'jumps':
        from .jump_channel import score
        return {**score(env.sim, sequence, target, basis, frame), 'duration_ns': float(duration)}
    if method in ('three_jumps', 'three_short'):
        if method == 'three_short':
            from .three_short_channel import score as fast_score, ConvergenceError

            def score(sim, sequence, target, basis, frame):
                try:
                    return fast_score(sim, sequence, target, basis, frame)
                except ConvergenceError:
                    from .three_jump_channel import score as full_score
                    return {**full_score(sim, sequence, target, basis, frame), 'fallback_reason': 'composite_zero_one_jump_quadrature_not_converged'}
        else:
            from .three_jump_channel import score
        from .parametric import enabled, CouplerSegment
        if enabled(env.dev) and all((not isinstance(x, Segment) or isinstance(x, CouplerSegment) for x in sequence)):
            from .three_idle_channel import restrict_number_conserving
            reduced, projected = restrict_number_conserving(env.sim, basis)
            return {**score(reduced, sequence, target, projected, frame), 'duration_ns': float(duration), 'reachable_hilbert_dimension': reduced.D}
        return {**score(env.sim, sequence, target, basis, frame), 'duration_ns': float(duration)}
    if method == 'three_idle':
        from .three_idle_channel import score
        return {**score(env.sim, sequence, target, basis, frame), 'duration_ns': float(duration)}
    if method == 'extended_batch':
        from .extended_dynamics import run_states
        d = len(target)
        eye = np.eye(d, dtype=complex)
        inputs = [np.outer(v, v.conj()) for v in eye]
        for i in range(d):
            for j in range(i + 1, d):
                for phase in (1.0, 1j):
                    v = (eye[i] + phase * eye[j]) / np.sqrt(2)
                    inputs.append(np.outer(v, v.conj()))
        states = run_states(env.sim, sequence, [qt.Qobj(basis @ rho @ basis.conj().T, dims=[env.sim.dims] * 2) for rho in inputs])
        outputs = []
        for state in states:
            out = basis.conj().T @ state.full() @ basis
            outputs.append(frame[:, None] * out * frame.conj()[None, :])
        iterator = iter(outputs)
        return {**average_channel(lambda rho: next(iterator), target), 'duration_ns': float(duration)}

    def evolve(rho):
        initial = qt.Qobj(basis @ rho @ basis.conj().T, dims=[env.sim.dims] * 2)
        _, state, _ = env.sim.run(sequence, rho0=initial, return_state=True)
        out = basis.conj().T @ state.full() @ basis
        return frame[:, None] * out * frame.conj()[None, :]
    result = average_channel(evolve, target)
    result['duration_ns'] = float(duration)
    return result

def decoherence_floor(dev, duration):
    d = 2 ** dev.n
    trace = np.prod([1 + 2 * np.exp(-duration * (0.5 / q.T1 + 1 / q.Tphi)) + np.exp(-duration / q.T1) for q in dev.qubits])
    return float(1 - (trace + d) / (d * (d + 1)))

def simultaneous_readout(env, k):
    env.sync_physics()
    from .hardware import temporal_enabled
    if temporal_enabled(env.dev):
        from .temporal_readout import score
        return score(env, k)
    n, cal = (env.dev.n, env.cal)
    means = [env.ro.iq_means(j, cal.readout_freq[j]) for j in range(n)]
    dur, t1 = (cal.readout_duration[k], env.sim.dev.qubits[k].T1)
    sigma = 0.15 / env.dev.resonators[k].snr_scale * np.sqrt(1000 / dur)
    from .instruments import receiver
    scale, offset, gain = receiver(env.dev, cal, k, dur)
    sigma *= gain
    angle, threshold = (cal.iq_angle[k], cal.iq_threshold[k])
    x, w = np.polynomial.legendre.leggauss(96)
    s = (x + 1) * dur / 2
    values = []
    for other in itertools.product((0, 1), repeat=n - 1):
        states = dict(zip([j for j in range(n) if j != k], other))
        cross = env.dev.readout_crosstalk
        leak = sum(((cross[k][j] if cross else 0) * (means[j][b] - means[j][0]) for j, b in states.items() if env.dev.resonators[j].feedline == env.dev.resonators[k].feedline))

        def p1(signal):
            return ndtr(((scale * (signal + leak) + offset) * np.exp(-1j * angle)).real / sigma - threshold / sigma)
        p0 = 1 - p1(means[k][0])
        excited = np.exp(-dur / t1) * p1(means[k][1]) + np.sum(w * dur / 2 * np.exp(-s / t1) / t1 * p1(s / dur * means[k][1] + (1 - s / dur) * means[k][0]))
        values.append({'neighbours': list(other), 'accuracy': float((p0 + excited) / 2)})
    return {'accuracy': min((v['accuracy'] for v in values)), 'contexts': values}

def specifications(env, family):
    n = env.dev.n
    for k in range(n):
        for name in ('x90', 'x180', 'id_gate', 'id_1us', 'readout'):
            category = 'readout' if name == 'readout' else 'idle' if name.startswith('id_') else 'single'
            yield {'id': f'q{k}/{name}', 'category': category, 'qubit': k, 'gate': name}
    if n > 1:
        for gates in itertools.product(('x90', 'x180'), repeat=n):
            yield {'id': 'parallel/' + '-'.join(gates), 'category': 'parallel', 'gates': list(gates)}
    for edge in env.dev.edges:
        for gate in ('cnot',) if family in ('cr', 'parametric_cnot') else ('cz', 'zz_idle'):
            yield {'id': f'edge{edge[0]}-{edge[1]}/{gate}', 'category': 'two_qubit' if gate != 'zz_idle' else 'idle', 'edge': list(edge), 'gate': gate}

def evaluate(env, family='cr', *, unknown=False, progress=lambda row: None, completed=None, metric_ids=None, stop_on_failure=False):
    env.sync_physics()
    policy = json.loads((Path(__file__).parents[1] / 'tasks/thresholds.json').read_text())
    rows = []
    idle_cache = {}
    saved = {r['id']: r for r in completed or []}
    specs = list(specifications(env, family))
    all_ids = {spec['id'] for spec in specs}
    if metric_ids is not None:
        metric_ids = list(metric_ids)
        if not metric_ids or len(set(metric_ids)) != len(metric_ids) or set(metric_ids) - all_ids:
            raise ValueError('Select distinct existing performance metric IDs')
        by_id = {spec['id']: spec for spec in specs}
        specs = [by_id[key] for key in metric_ids]
    for spec in specs:
        if spec['id'] in saved:
            cached = saved[spec['id']]
            if any((cached.get(k) != v for k, v in spec.items())) or cached['threshold'] != policy[spec['category']]:
                raise ValueError('Saved performance context differs from the current specification')
            rows.append(deepcopy(cached))
            progress(rows[-1])
            continue
        row = dict(spec, value=None, passed=None, error=None, threshold=policy[spec['category']])
        if not unknown:
            try:
                if spec['category'] == 'readout':
                    result = simultaneous_readout(env, spec['qubit'])
                    error = 1 - result['accuracy']
                elif spec['category'] == 'parallel':
                    samples = []
                    for phases in itertools.product((0.0, np.pi / 2, np.pi, 3 * np.pi / 2), repeat=env.dev.n - 1):
                        phases = (0.0,) + phases
                        angles = [np.pi / 2 if g == 'x90' else np.pi for g in spec['gates']]
                        seq = [Segment([env.x_pulse(k, a, phase=phases[k]) for k, a in enumerate(angles)])]
                        target = product([rotation(a, p) for a, p in zip(angles, phases)])
                        samples.append(channel_score(env, seq, target))
                    result = {'fidelity': float(np.mean([v['fidelity'] for v in samples])), 'phase_samples': len(samples), 'worst_phase_fidelity': min((v['fidelity'] for v in samples))}
                    if any(('fidelity_interval' in v for v in samples)):
                        result['fidelity_interval'] = np.mean([v.get('fidelity_interval', [v['fidelity']] * 2) for v in samples], axis=0).tolist()
                        result['value_kind'] = 'conservative_fidelity_lower_bound'
                    error = 1 - result['fidelity']
                else:
                    gate = spec['gate']
                    if 'edge' in spec:
                        seq, phases = compiled(env, gate, tuple(spec['edge']))
                        target = np.eye(2 ** env.dev.n) if gate == 'zz_idle' else two_target(env.dev.n, spec['edge'], gate)
                    elif gate.startswith('id_'):
                        duration = 1000.0 if gate == 'id_1us' else env.cal.pi_duration[spec['qubit']]
                        seq, phases = installed_idle(env, duration)
                        target = np.eye(2 ** env.dev.n)
                    else:
                        k = spec['qubit']
                        angle = np.pi / 2 if gate == 'x90' else np.pi
                        seq, phases = ([Segment([env.x_pulse(k, angle)])], None)
                        target = product([rotation(angle) if j == k else np.eye(2) for j in range(env.dev.n)])
                    key = ('installed_idle', duration) if gate.startswith('id_') else seq[0] if len(seq) == 1 and isinstance(seq[0], float) else None
                    result = idle_cache.get(key) if key is not None else None
                    if result is None:
                        result = channel_score(env, seq, target, phases)
                        if key is not None:
                            idle_cache[key] = result
                    result = dict(result)
                    error = 1 - result['fidelity']
                    if spec['category'] == 'idle':
                        result['decoherence_floor'] = decoherence_floor(env.sim.dev, result['duration_ns'])
                        error = max(0.0, error - result['decoherence_floor'])
                if not np.isfinite(error):
                    raise ValueError('Non-finite performance')
                row.update(value=result, error=float(error), passed=bool(error <= row['threshold']))
            except (KeyError, ValueError) as exc:
                row.update(unavailable=str(exc), passed=False)
        rows.append(row)
        progress(row)
        if stop_on_failure and row['passed'] is False and (row['error'] is not None):
            break
    complete = {row['id'] for row in rows} == all_ids
    return {'schema_version': 2, 'solver_mode': env.sim.mode, 'definition': 'Full computational-space Haar average with leakage; readout worst neighbour-conditioned balanced accuracy', 'complete': complete, 'passed': False if any((r['passed'] is False for r in rows)) else all((r['passed'] is True for r in rows)) if complete else None, 'metrics': rows}
