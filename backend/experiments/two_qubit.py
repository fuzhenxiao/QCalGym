from __future__ import annotations
from typing import Optional, List
import numpy as np
from scipy.optimize import least_squares
from ..core import Pulse, Segment
from ..env import Env
from .. import fitting as F
from .common import Result, new_fig
from .single import ramsey
from .figures import ramsey_figure
TWO_PI = 2 * np.pi

def conditional_ramsey(env: Env, k: int, spectator: int, delays: np.ndarray, virtual_detuning: float=0.002, shots: int=200) -> Result:
    r0 = ramsey(env, k, delays, virtual_detuning, shots, spectator=None)
    r1 = ramsey(env, k, delays, virtual_detuning, shots, spectator={spectator: 1})
    ok = r0.fit['status'] == r1.fit['status'] == 'finished'
    zz = r0.fit['params'].get('freq', np.nan) - r1.fit['params'].get('freq', np.nan) if ok else np.nan
    fit = {'status': 'finished' if ok else 'failed', 'model': 'two_ramsey', 'r2': min(r0.fit['r2'], r1.fit['r2']), 'params': {'zz_GHz': float(zz), 'f_spec0': r0.fit['params'].get('freq'), 'f_spec1': r1.fit['params'].get('freq')}, 'errors': {}}
    fig = ramsey_figure(f'Conditional Ramsey q{k} | spectator q{spectator}: ZZ={zz * 1000000.0:.1f} kHz', delays, [r0.points['p1'], r1.points['p1']], [r0.fit, r1.fit], ['spectator |0>', 'spectator |1>'])
    for r in (r0, r1):
        r.figure and __import__('matplotlib.pyplot').pyplot.close(r.figure)
    return Result('conditional_ramsey', {'delay_ns': delays, 'p1_spec0': r0.points['p1'], 'p1_spec1': r1.points['p1']}, fit, {'qubit': k, 'spectator': spectator, 'shots': shots}, fig)

def cr_pulse(env: Env, control: int, target: int, amp: Optional[float]=None, duration: Optional[float]=None, phase: Optional[float]=None, rise: Optional[float]=None) -> Pulse:
    return env.cr_pulse(control, target, amp, duration, phase, rise)

def cr_amp_scan(env: Env, control: int, target: int, amps: np.ndarray, duration: Optional[float]=None, phase: Optional[float]=None, shots: int=200) -> Result:
    duration = env.cal.cr_duration.get((control, target), 300.0) if duration is None else duration
    p1 = {0: [], 1: []}
    sequences = []
    branches = []
    for a in amps:
        for ctrl in (0, 1):
            seq = ([Segment([env.x_pulse(control)])] if ctrl else []) + [env.cr_segment(control, target, amp=float(a), duration=duration, phase=phase)]
            sequences.append(seq)
            branches.append(ctrl)
    for ctrl, iq in zip(branches, env.measure_many(sequences, shots, [target, control], name='cr_amp')):
        p1[ctrl].append(env.p1(iq[target], target))
    p0 = np.array(p1[0])
    p1_ = np.array(p1[1])
    f0 = F.fit_cosine(amps, p0)
    f1 = F.fit_cosine(amps, p1_)
    fit = {'status': 'finished' if f0['status'] == f1['status'] == 'finished' else 'failed', 'model': 'two_cosines', 'r2': min(f0['r2'], f1['r2']), 'params': {'rate_ctrl0_per_amp': f0['params'].get('freq'), 'rate_ctrl1_per_amp': f1['params'].get('freq')}, 'errors': {}}
    fig, ax = new_fig(f'CR amp scan c{control}->t{target}, {duration:.0f} ns', 'CR amplitude (a.u.)', 'target P(1)')
    ax.plot(amps, p0, 'o', ms=3, label='control |0>')
    ax.plot(amps, p1_, 's', ms=3, label='control |1>')
    if f0['status'] == 'finished':
        ax.plot(amps, f0['yfit'], '-', lw=1)
    if f1['status'] == 'finished':
        ax.plot(amps, f1['yfit'], '-', lw=1)
    ax.legend(fontsize=8)
    return Result('cr_amp_scan', {'amp': amps, 'p1_ctrl0': p0, 'p1_ctrl1': p1_}, fit, {'control': control, 'target': target, 'duration': duration, 'shots': shots}, fig)

def _bloch_traj(t, wx, wy, dz):
    W = np.sqrt(wx ** 2 + wy ** 2 + dz ** 2) + 1e-15
    n = np.array([wx, wy, dz]) / W
    r0 = np.array([0.0, 0.0, 1.0])
    c = np.cos(W * t)[:, None]
    s = np.sin(W * t)[:, None]
    cross = np.cross(n, r0)
    return r0 * c + cross * s + n * (n @ r0) * (1 - c)

def _fit_bloch(t, X, Y, Z):

    def resid(p):
        r = _bloch_traj(t, *p)
        return np.concatenate([r[:, 0] - X, r[:, 1] - Y, r[:, 2] - Z])
    best = None
    f_guess = F._guess_freq(t, Z)
    for w0 in [TWO_PI * f_guess, TWO_PI * f_guess / 2, 0.005]:
        for ang in np.linspace(0, TWO_PI, 6, endpoint=False):
            for dz0 in (0.0, w0 / 2, -w0 / 2):
                p0 = [w0 * np.cos(ang), w0 * np.sin(ang), dz0]
                try:
                    sol = least_squares(resid, p0)
                except Exception:
                    continue
                if best is None or sol.cost < best.cost:
                    best = sol
    return best

def cr_hamiltonian_tomography(env: Env, control: int, target: int, durations: np.ndarray, amp: Optional[float]=None, phase: Optional[float]=None, shots: int=200, rise: Optional[float]=None) -> Result:
    amp = env.cal.cr_amp[control, target] if amp is None else amp
    tomo = {'X': lambda: env.y_pulse(target, np.pi / 2), 'Y': lambda: env.x_pulse(target, np.pi / 2), 'Z': None}
    data = {}
    if env.dev.n <= 2:
        sequences = []
        keys = []
        for ctrl in (0, 1):
            for basis, pre in tomo.items():
                data[ctrl, basis] = []
                for d in durations:
                    seq = [Segment([env.x_pulse(control)])] if ctrl else []
                    seq += [env.cr_segment(control, target, amp=amp, duration=float(d), phase=phase, rise=rise)]
                    if pre is not None:
                        seq += [Segment([pre()])]
                    sequences.append(seq)
                    keys.append((ctrl, basis))
        for key, iq in zip(keys, env.measure_many(sequences, shots, [target], name='cr_tomo')):
            data[key].append(1 - 2 * env.p1(iq[target], target))
        data = {key: np.asarray(values) for key, values in data.items()}
    else:
        for ctrl in (0, 1):
            for basis, pre in tomo.items():
                vals = []
                for d in durations:
                    seq = [Segment([env.x_pulse(control)])] if ctrl else []
                    seq += [env.cr_segment(control, target, amp=amp, duration=float(d), phase=phase, rise=rise)]
                    if pre is not None:
                        seq += [Segment([pre()])]
                    iq = env.measure(seq, shots, [target], name='cr_tomo')[target]
                    vals.append(1 - 2 * env.p1(iq, target))
                data[ctrl, basis] = np.array(vals)
    sols = {}
    for ctrl in (0, 1):
        sols[ctrl] = _fit_bloch(durations, data[ctrl, 'X'], data[ctrl, 'Y'], data[ctrl, 'Z'])
    ok = all((s is not None for s in sols.values()))
    params = {}
    if ok:
        w0, w1 = (sols[0].x, sols[1].x)
        for name, i in zip(('X', 'Y', 'Z'), range(3)):
            params[f'I{name}_GHz'] = float((w1[i] + w0[i]) / 2 / TWO_PI)
            params[f'Z{name}_GHz'] = float((w1[i] - w0[i]) / 2 / TWO_PI)
        params['w_ctrl0_rad_per_ns'] = w0.tolist()
        params['w_ctrl1_rad_per_ns'] = w1.tolist()
    fit = {'status': 'finished' if ok else 'failed', 'model': 'bloch_rotation', 'params': params, 'errors': {}, 'r2': float('nan')}
    fig, axes = __import__('matplotlib.pyplot').pyplot.subplots(3, 1, figsize=(5.2, 6.5), sharex=True)
    for ax, basis in zip(axes, 'XYZ'):
        for ctrl, mk in ((0, 'o'), (1, 's')):
            ax.plot(durations, data[ctrl, basis], mk, ms=3, label=f'control |{ctrl}>')
            if ok:
                r = _bloch_traj(durations, *sols[ctrl].x)
                ax.plot(durations, r[:, 'XYZ'.index(basis)], '-', lw=1)
        ax.set_ylabel(f'<{basis}>')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('CR duration (ns)')
    axes[0].set_title(f'CR Hamiltonian tomography c{control}->t{target}, amp={amp}', fontsize=10)
    return Result('cr_hamiltonian_tomography', {'duration_ns': durations, **{f'{b}_ctrl{c}': v for (c, b), v in data.items()}}, fit, {'control': control, 'target': target, 'amp': amp, 'shots': shots}, fig)
