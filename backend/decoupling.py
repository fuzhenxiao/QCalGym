from dataclasses import replace
import numpy as np
from .core import Pulse, Segment

def idle(env, duration, frames=None):
    from .hardware import validate_controls
    from .gates import installed_idle
    validate_controls(env.dev, env.cal)
    if env.dev.n > 2:
        raise ValueError('DD currently supports one/pair apparatus')
    seq, advance = installed_idle(env, duration, frames, _skip_dd=True)
    carried = np.zeros(env.dev.n) if frames is None else np.asarray(frames)
    pulses, clock = ([], 0.0)
    for el in seq:
        if isinstance(el, Segment):
            pulses.extend((replace(p, start=p.start + clock) for p in el.pulses))
            clock += el.duration
        else:
            clock += el
    added = False
    for k in range(env.dev.n):
        mode = env.cal.dd_sequence.get(k, 'off')
        if mode == 'off' or duration < env.cal.dd_min_duration.get(k, 200.0):
            continue
        phases = [0.0] * env.cal.dd_pulses.get(k, 2) if mode == 'cpmg' else [0.0, np.pi / 2, 0.0, np.pi / 2] if mode == 'xy4' else [0.0, np.pi / 2, 0.0, np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0]
        spacing = duration / len(phases)
        width = env.cal.pi_duration[k]
        starts = [(j + 0.5 + env.cal.dd_offset.get(k, 0.0)) * spacing - width / 2 for j in range(len(phases))]
        if width > spacing or starts[0] < 0 or starts[-1] + width > duration:
            raise ValueError('Finite DD pulses do not fit the requested idle window')
        pulses.extend((replace(env.x_pulse(k, phase=p + carried[k]), start=s) for p, s in zip(phases, starts)))
        added = True
    if not added:
        return (seq, advance)
    pulses.append(Pulse(0, env.cal.drive_freq[0], 0.0, duration, shape='square'))
    return ([Segment(pulses)], advance)
