import itertools
import numpy as np
from scipy.special import ndtr

def photon_fields(res, frequency, times, drive_duration, photon_flux):
    times = np.asarray(times, float)
    if photon_flux < 0 or drive_duration < 0 or np.any(times < 0):
        raise ValueError('Times, duration and photon flux must be nonnegative')
    shifts = np.asarray(res.chi_levels if res.chi_levels else [0, res.chi])[:2]
    gamma = np.pi * res.kappa + 2j * np.pi * (frequency - res.freq - shifts)
    drive = np.sqrt(2 * np.pi * res.kappa * photon_flux)
    on_time = np.minimum(times, drive_duration)
    return drive / gamma[:, None] * -np.expm1(-gamma[:, None] * on_time) * np.exp(-gamma[:, None] * np.maximum(times - drive_duration, 0))

def integration_weights(values, bins):
    if values is None:
        return np.ones(bins, complex)
    a = np.asarray(values, float)
    if a.ndim != 2 or a.shape[1] != 2 or (not len(a)) or (not np.isfinite(a).all()) or (not np.any(a)):
        raise ValueError('Invalid complex readout weights')
    left = np.maximum(np.arange(bins)[:, None] / bins, np.arange(len(a))[None, :] / len(a))
    right = np.minimum((np.arange(bins)[:, None] + 1) / bins, (np.arange(len(a))[None, :] + 1) / len(a))
    out = np.einsum('ij,j->i', np.maximum(right - left, 0) * bins, a[:, 0] + 1j * a[:, 1], optimize=False)
    if not np.any(np.abs(out) > 1e-14):
        raise ValueError('Readout weights cancel every acquisition bin')
    return out

def waveforms(res, frequency, labels, jumps, edges, drive_duration):
    labels = np.asarray(labels, int)
    jumps = np.asarray(jumps, float)
    beta = np.zeros(len(labels), complex)
    result = np.empty((len(labels), len(edges) - 1), complex)
    shifts = np.asarray(res.chi_levels if res.chi_levels else [res.chi * m for m in range(3)])
    for col, (left, right) in enumerate(zip(edges, edges[1:])):
        cuts = np.sort(np.column_stack((np.full(len(labels), left), np.clip(jumps[:, 0], left, right), np.clip(jumps[:, 1], left, right), np.full(len(labels), np.clip(drive_duration, left, right)), np.full(len(labels), right))), axis=1)
        integrated = np.zeros(len(labels), complex)
        for a, b in zip(cuts.T[:-1], cuts.T[1:]):
            dt = b - a
            mid = (a + b) / 2
            level = np.maximum(0, labels - (jumps[:, 0] <= mid) - (jumps[:, 1] <= mid))
            gamma = np.pi * res.kappa + 2j * np.pi * (frequency - res.freq - shifts[level])
            on = (mid < drive_duration).astype(float)
            steady = np.pi * 0.8 * res.kappa * on / gamma
            decay = np.exp(-gamma * dt)
            integral_beta = steady * dt + (beta - steady) * -np.expm1(-gamma * dt) / gamma
            integrated += on * dt - integral_beta
            beta = steady + (beta - steady) * decay
        result[:, col] = integrated / (right - left)
    return result

def sample(chain, pops, frequencies, shots, qubits, durations):
    n, dev, rng = (chain.dev.n, chain.dev, chain.rng)
    flat = np.clip(np.asarray(pops).ravel(), 0, None)
    labels = np.array(np.unravel_index(rng.choice(len(flat), shots, p=flat / flat.sum()), pops.shape)).T
    jumps = []
    for j, q in enumerate(dev.qubits):
        first = np.full(shots, np.inf)
        active = labels[:, j] > 0
        first[active] = rng.exponential(q.T1 / labels[active, j])
        second = np.full(shots, np.inf)
        two = labels[:, j] == 2
        second[two] = first[two] + rng.exponential(q.T1, np.sum(two))
        jumps.append(np.column_stack((first, second)))
    bins = dev.physics['readout'].get('bins', 64)
    out = {}
    for k in qubits:
        edges = np.linspace(0, durations[k], bins + 1)
        signal = chain.waveforms(k, frequencies[k], labels[:, k], jumps[k], edges, durations[k])
        cross = dev.readout_crosstalk
        for j in range(n):
            if j == k or not cross or (not cross[k][j]) or (dev.resonators[j].feedline != dev.resonators[k].feedline):
                continue
            neighbour = chain.waveforms(j, frequencies[j], labels[:, j], jumps[j], edges, durations[j])
            ground = chain.waveforms(j, frequencies[j], [0], [[np.inf, np.inf]], edges, durations[j])
            signal += cross[k][j] * (neighbour - ground)
        sigma = 0.15 / dev.resonators[k].snr_scale * np.sqrt(1000 / (durations[k] / bins))
        from .instruments import receiver
        scale, offset, gain = receiver(dev, chain.electronics_cal, k, durations[k])
        out[k] = scale * signal + offset + gain * sigma * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return out

def integrate(traces, weights):
    return {k: np.einsum('ij,j->i', v, integration_weights(weights.get(k), v.shape[1]).conj(), optimize=False) / v.shape[1] for k, v in traces.items()}

def _components(chain, k, frequency, state, t1, edges, drive_duration, weight, order):
    end = edges[-1]
    if not state or not np.isfinite(t1):
        jumps, probability = (np.array([[np.inf, np.inf]]), np.array([1.0]))
    else:
        x, w = np.polynomial.legendre.leggauss(order)
        times = (x + 1) * end / 2
        jumps = np.column_stack((np.r_[np.inf, times], np.full(order + 1, np.inf)))
        probability = np.r_[np.exp(-end / t1), w * end / 2 * np.exp(-times / t1) / t1]
    waves = chain.waveforms(k, frequency, np.full(len(jumps), state), jumps, edges, drive_duration)
    return (np.einsum('ij,j->i', waves, weight.conj(), optimize=False) / len(weight), probability)

def score(env, k):
    dev, cal = (env.ro.dev, env.cal)
    bins = dev.physics['readout'].get('bins', 64)
    duration = cal.readout_duration[k]
    edges = np.linspace(0, duration, bins + 1)
    weight = integration_weights(cal.readout_weights.get(k), bins)
    sigma = 0.15 / dev.resonators[k].snr_scale * np.sqrt(1000 / duration) * np.linalg.norm(weight) / np.sqrt(bins)
    from .instruments import receiver
    scale, offset, gain = receiver(dev, cal, k, duration)
    sigma *= gain
    offset *= weight.conj().mean()
    if sigma == 0:
        raise ValueError('Readout weights cancel every acquisition bin')
    previous = None
    for order in (24, 48, 96, 192):
        rows = []
        for other in itertools.product((0, 1), repeat=dev.n - 1):
            accuracy = []
            for state in (0, 1):
                signal, probability = _components(env.ro, k, cal.readout_freq[k], state, dev.qubits[k].T1, edges, duration, weight, order)
                for j, b in zip([j for j in range(dev.n) if j != k], other):
                    cross = dev.readout_crosstalk
                    if not cross or not cross[k][j] or dev.resonators[k].feedline != dev.resonators[j].feedline:
                        continue
                    neighbour, prob = _components(env.ro, j, cal.readout_freq[j], b, dev.qubits[j].T1, edges, cal.readout_duration[j], weight, order)
                    ground, _ = _components(env.ro, j, cal.readout_freq[j], 0, dev.qubits[j].T1, edges, cal.readout_duration[j], weight, order)
                    signal = (signal[:, None] + cross[k][j] * (neighbour[None, :] - ground[0])).ravel()
                    probability = (probability[:, None] * prob[None, :]).ravel()
                signal = scale * signal + offset
                p1 = ndtr(((signal * np.exp(-1j * cal.iq_angle[k])).real - cal.iq_threshold[k]) / sigma)
                accuracy.append(float(np.sum(probability * (p1 if state else 1 - p1))))
            rows.append({'neighbours': list(other), 'accuracy': float(np.mean(accuracy))})
        current = np.array([r['accuracy'] for r in rows])
        error = np.inf if previous is None else float(np.max(np.abs(current - previous)))
        if error < 2e-06:
            return {'accuracy': float(current.min()), 'contexts': rows, 'model': 'time_resolved_cavity', 'quadrature_order': order, 'accuracy_change': error}
        previous = current
    raise ValueError(f'Time-readout jump quadrature not converged: change={error:.3g}')
