from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import numpy as np
import qutip as qt

def supported(dev):
    return dev.n == 3 and dev.topology in ('chain3', 'ring3')

def ramsey_traces(env, k, delays, branches, virtual_detuning, shots):
    from .core import Segment
    from . import fitting
    sequences, names = ([], [])
    for tones, spectator, name in branches:
        prep = [Segment([env.x_pulse(j) for j, level in spectator.items() if level == 1])] if spectator else []
        for tau in delays:
            sequence = prep + [Segment([env.x_pulse(k, np.pi / 2)])]
            if tau > 0:
                sequence.append(Segment([env.drive_pulse(line, f, amp, float(tau) + 120.0, phase=phase, shape='gaussian_square', rise=60.0) for line, f, amp, phase in tones]))
            sequence.append(Segment([env.x_pulse(k, np.pi / 2, phase=2 * np.pi * virtual_detuning * tau)]))
            sequences.append(sequence)
            names.append(name)
    values = []
    results = env.sim.run_many(sequences, return_state=True)
    try:
        for name, (pops, rho, end) in zip(names, results):
            iq = env._sample_simulation(pops, end, shots, [k], name)
            values.append(env.p1(iq[k], k))
    finally:
        results.close()
    return [fitting.fit_damped_cosine(delays, np.asarray(values[start:start + len(delays)])) for start in range(0, len(values), len(delays))]

def pulse(sim, rho, segment, t0):
    duration = segment.duration
    rise = segment.pulses[0].rise
    w = 2 * np.pi * segment.pulses[0].freq
    initial = sim.to_frame(rho, t0, w)
    descriptors = []
    for p in segment.pulses:
        descriptor = asdict(p)
        descriptor.pop('duration')
        descriptors.append(descriptor)
    key = (json.dumps(descriptors, sort_keys=True), float(t0), json.dumps(sim.opts, sort_keys=True), hashlib.sha256(initial.full().tobytes()).digest())
    cache = getattr(sim, '_three_plateaus', None)
    if cache is None:
        cache = sim._three_plateaus = OrderedDict()
    h = [sim._static_frame_H(w)] + sim._drive_terms(segment, t0, w)
    options = {**sim._qutip_segment_options(duration), 'normalize_output': False}
    options['max_step'] = min(options.get('max_step', 0) or rise / 4, rise / 4)
    entry = cache.get(key)
    plateau = duration - 2 * rise
    if entry is None:
        first = qt.mesolve(h, initial, [t0, t0 + rise], c_ops=sim.c_ops, options=options).final_state
        entry = {'first': first, 'last': first, 'time': 0.0, 'hamiltonian': sim._constant_H(segment, w)}
        cache[key] = entry
        while len(cache) > 32:
            cache.popitem(last=False)
    cache.move_to_end(key)
    previous = entry['time'] if entry['time'] <= plateau else 0.0
    middle = entry['last'] if previous else entry['first']
    if plateau > previous:
        middle = qt.mesolve(entry['hamiltonian'], middle, [0.0, plateau - previous], c_ops=sim.c_ops, options=options).final_state
    if plateau >= entry['time']:
        entry.update(time=plateau, last=middle)
    final = qt.mesolve(h, middle, [t0 + duration - rise, t0 + duration], c_ops=sim.c_ops, options=options).final_state
    return sim.to_lab(final, t0 + duration, w)
