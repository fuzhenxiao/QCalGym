import numpy as np
from .core import Segment

def rotation(angle, phase=0.0):
    axis = np.array([[0, np.exp(1j * phase)], [np.exp(-1j * phase), 0]])
    return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * axis

def product(ops):
    result = np.ones((1, 1), complex)
    for op in ops:
        result = np.kron(result, op)
    return result

def two_target(n, edge, kind):
    out = np.zeros((2 ** n, 2 ** n), complex)
    c, t = edge
    for col in range(2 ** n):
        bits = [col >> n - 1 - k & 1 for k in range(n)]
        sign = -1 if kind == 'cz' and bits[c] and bits[t] else 1
        if kind == 'cnot':
            bits[t] ^= bits[c]
        row = sum((b << n - 1 - k for k, b in enumerate(bits)))
        out[row, col] = sign
    return out

def compiled(env, kind, edge, frames=None, *, _skip_dd=False):
    if 'flux_coupler' in env.dev.physics:
        from .flux_gates import compiled as flux_compiled
        return flux_compiled(env, kind, edge, frames)
    if kind == 'zz_idle' and (not _skip_dd) and any((v != 'off' for v in env.cal.dd_sequence.values())):
        return installed_idle(env, 1000.0, frames)
    from .parametric import enabled, compiled as parametric_compiled
    if enabled(env.dev):
        return parametric_compiled(env, kind, edge, frames)
    c, t = edge
    phases = np.zeros(env.dev.n)
    carried = np.zeros(env.dev.n) if frames is None else np.asarray(frames)
    if kind == 'zz_idle' and env.cal.idle_active.get(0, 0):
        return network_idle_block(env)
    if kind == 'cnot':
        pre = np.array([env.cal.cnot_pre_phase.get(k, 0.0) for k in range(env.dev.n)])
        if pre.any():
            carried = carried + pre
            phases += pre
        if env.cal.ecr_walsh.get(edge, 0):
            seq = walsh_ecr(env, edge, carried)
            phases[c] -= np.pi / 2
            phases += env.cal.post_gate_network_phase.get(edge, [0.0] * env.dev.n)
            return (seq, phases)
        seq = env.ecr_sequence(c, t)
        for index in (1, 3):
            seq[index].pulses.extend((env.x_pulse(k) for k in range(env.dev.n) if k not in edge))
        seq.append(Segment([env.x_pulse(t, -np.pi / 2)]))
        for index in (0, 2):
            for pulse in seq[index].pulses:
                pulse.phase += carried[t]
        for index in (1, 3):
            for pulse in seq[index].pulses:
                pulse.phase += carried[pulse.line]
        seq[-1].pulses[0].phase += carried[t]
        phases[c] -= np.pi / 2
        if edge not in env.cal.post_gate_network_phase:
            for k in range(env.dev.n):
                phases[k] += env.cal.cnot_frame_phase.get(k, 0.0)
    elif kind in ('cz', 'zz_idle'):
        if kind == 'zz_idle':
            cal = env.cal.copy()
            for field in ('freq', 'amp', 'phase', 'rise'):
                getattr(cal, 'sizzle_' + field)[edge] = getattr(cal, 'hold_' + field)[edge]
            cal.sizzle_smooth[edge] = cal.hold_smooth.get(edge, 0)
            from .env import Env
            hold = Env(env.dev, cal, solver_mode=env.sim.mode)
            seq = [hold.sizzle_segment(c, t, duration=1000.0)]
            phases[list(edge)] = env.cal.hold_frame.get(edge, (0.0, 0.0))
            return (seq, phases)
        if edge in env.cal.sizzle_network_amp:
            duration, rise = (env.cal.sizzle_duration[edge], env.cal.sizzle_rise[edge])
            if 2 * rise > duration:
                raise ValueError('Network CZ duration must be at least twice rise')
            seq = [Segment([env.drive_pulse(k, env.cal.sizzle_freq[edge], amp, duration, phase=env.cal.sizzle_network_phase[edge][k], rise=rise, shape='gaussian_square', smooth_edges=bool(env.cal.sizzle_smooth.get(edge, 0))) for k, amp in enumerate(env.cal.sizzle_network_amp[edge])])]
        else:
            seq = [env.sizzle_segment(c, t)]
    else:
        raise ValueError('Unknown installed gate')
    if edge in env.cal.post_gate_network_phase:
        phases += env.cal.post_gate_network_phase[edge]
    else:
        phases[list(edge)] += env.cal.post_gate_phase.get(edge, (0.0, 0.0))
    return (seq, phases)

def walsh_ecr(env, edge, carried):
    c, t = edge
    cal = env.cal
    quarter = cal.cr_duration[edge] / 4
    rise = cal.cr_rise[edge]
    if quarter < 2 * rise:
        raise ValueError('Walsh CR quarter must be at least twice cr_rise')
    spectators = [k for k in range(env.dev.n) if k not in edge]
    if len(spectators) > 2:
        raise ValueError('Walsh compiler supports at most two spectators')
    signs = {c: [1, 1, -1, -1], **dict(zip(spectators, [[1, -1, 1, -1], [1, -1, -1, 1]]))}
    sequence = []
    for j in range(4):
        second = j >= 2
        phase = cal.cr_phase[edge] + (np.pi + cal.ecr_phase_skew.get(edge, 0.0) if second else 0.0) + carried[t]
        amp = cal.cr_amp[edge] * (1 + (-1 if second else 1) * cal.ecr_balance.get(edge, 0.0))
        sequence.append(env.cr_segment(c, t, amp=amp, duration=quarter, phase=phase, rotary=True))
        flipped = [k for k, pattern in signs.items() if pattern[j] != (pattern[j + 1] if j < 3 else 1)]
        if flipped:
            sequence.append(Segment([env.x_pulse(k, phase=carried[k]) for k in flipped]))
    sequence.append(Segment([env.x_pulse(t, -np.pi / 2, phase=carried[t])]))
    return sequence

def network_idle_block(env):
    cal = env.cal
    for k in range(env.dev.n):
        if not 0 < cal.idle_tone_rise[k] <= 500:
            raise ValueError('Network idle rise must be in (0,500] ns')
    pulses = [env.drive_pulse(k, cal.idle_tone_freq[k], cal.idle_tone_amp[k], 1000.0, phase=cal.idle_tone_phase[k], rise=cal.idle_tone_rise[k], shape='gaussian_square', smooth_edges=bool(cal.idle_tone_smooth.get(k, 0))) for k in range(env.dev.n)]
    return ([Segment(pulses)], np.array([cal.idle_frame_phase[k] for k in range(env.dev.n)]))

def installed_idle(env, duration, frames=None, *, _skip_dd=False):
    carried = np.zeros(env.dev.n) if frames is None else np.asarray(frames).copy()
    if not np.isfinite(duration) or duration < 0:
        raise ValueError('Idle duration must be finite and nonnegative')
    if not _skip_dd and any((v != 'off' for v in env.cal.dd_sequence.values())):
        from .decoupling import idle
        return idle(env, duration, frames)
    from .parametric import enabled
    if enabled(env.dev):
        return ([float(duration)], np.zeros(env.dev.n))
    enabled = [e for e in env.dev.edges if env.cal.hold_enabled.get(e, 0)]
    network = env.cal.idle_active.get(0, 0)
    if not enabled and (not network):
        return ([float(duration)], np.zeros(env.dev.n))
    if not network and (env.dev.n != 2 or len(enabled) != 1):
        raise ValueError('Installed compensated idle is currently defined for pair devices only')
    count, remainder = divmod(float(duration), 1000.0)
    seq = []
    total = np.zeros(env.dev.n)
    for _ in range(int(count)):
        block, advance = network_idle_block(env) if network else compiled(env, 'zz_idle', enabled[0], frames=carried, _skip_dd=True)
        seq.extend(block)
        carried += advance
        total += advance
    if remainder or not seq:
        seq.append(remainder)
    return (seq, total)
