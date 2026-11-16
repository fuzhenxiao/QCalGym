from __future__ import annotations
from typing import Optional, List
import numpy as np
from ..core import Pulse, Segment
from ..env import Env
from .. import fitting as F
from .common import Result, new_fig, plot_fit
from .figures import iq_figure, ramsey_figure

def resonator_spec(env: Env, k: int, freqs: np.ndarray, n_avg: int=200, excited: bool=False) -> Result:
    env.sync_physics()
    seq = [Segment([env.x_pulse(k)])] if excited else []
    pops = env.sim.run(seq) if seq else env.sim.populations(env.sim.ground())
    mag = env.ro.resonator_spec(k, freqs, env.sim.marginal(pops, k), n_avg=n_avg)
    env.ledger.add('resonator_spec', n_avg * len(freqs), env.cal.readout_duration.get(k, 1000.0) + env.readout_overhead_ns, qubit=k)
    fit = F.fit_lorentzian(freqs, mag, dip=True)
    fig, ax = new_fig(f'Resonator spectroscopy q{k}' + (' (qubit in |1>)' if excited else ''), 'frequency (GHz)', '|S21|')
    plot_fit(ax, freqs, mag, fit, data_line=True)
    return Result('resonator_spec', {'freq_GHz': freqs, 's21_mag': mag}, fit, {'qubit': k, 'n_avg': n_avg}, fig)

def qubit_spec(env: Env, k: int, freqs: np.ndarray, amp: float=0.02, duration: float=2000.0, shots: int=200, shape: str='gaussian_square') -> Result:
    iq_mean, p1 = ([], [])
    sequences = [[Segment([env.drive_pulse(k, f, amp, duration, shape=shape, rise=duration / 5)])] for f in freqs]
    for result in env.measure_many(sequences, shots, [k], name='qubit_spec'):
        iq = result[k]
        iq_mean.append(iq.mean())
        p1.append(env.p1(iq, k))
    iq_mean = np.array(iq_mean)
    p1 = np.array(p1)
    n_edge = max(2, len(freqs) // 10)
    base = np.concatenate([iq_mean[:n_edge], iq_mean[-n_edge:]]).mean()
    resp = np.abs(iq_mean - base)
    fit = F.fit_lorentzian(freqs, resp, dip=False)
    fig, ax = new_fig(f'Qubit spectroscopy q{k} (amp={amp}, {duration:.0f} ns)', 'drive frequency (GHz)', '|IQ response|')
    plot_fit(ax, freqs, resp, fit)
    return Result('qubit_spec', {'freq_GHz': freqs, 'response': resp, 'p1': p1, 'iq_mean': iq_mean}, fit, {'qubit': k, 'amp': amp, 'duration': duration, 'shots': shots}, fig)

def rabi_time(env: Env, k: int, durations: np.ndarray, amp: Optional[float]=None, shots: int=200, rise: float=5.0) -> Result:
    amp = env.cal.pi_amp.get(k, 0.5) * 0.4 if amp is None else amp
    p1, iq_mean = ([], [])
    sequences = [[] if d <= 0 else [Segment([env.drive_pulse(k, env.cal.drive_freq[k], amp, float(d), shape='gaussian_square', rise=rise)])] for d in durations]
    for result in env.measure_many(sequences, shots, [k], name='rabi_time'):
        p1.append(env.p1(result[k], k))
        iq_mean.append(result[k].mean())
    p1 = np.array(p1)
    fit = F.fit_damped_cosine(durations, p1)
    meta = {'qubit': k, 'amp': amp, 'shots': shots}
    fig, ax = new_fig(f'Rabi (time) q{k}, amp={amp:.3f}', 'pulse duration (ns)', 'P(1)')
    plot_fit(ax, durations, p1, fit)
    return Result('rabi_time', {'duration_ns': durations, 'p1': p1, 'iq_mean': np.asarray(iq_mean)}, fit, meta, fig)

def rabi_amp(env: Env, k: int, amps: np.ndarray, duration: Optional[float]=None, shots: int=200) -> Result:
    duration = env.cal.pi_duration.get(k, 40.0) if duration is None else duration
    p1 = []
    sequences = [[Segment([env.x_pulse(k, amp=float(a), duration=duration)])] for a in amps]
    for result in env.measure_many(sequences, shots, [k], name='rabi_amp'):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    fit = F.fit_cosine(amps, p1)
    meta = {'qubit': k, 'duration': duration, 'shots': shots}
    fig, ax = new_fig(f'Rabi (amplitude) q{k}, {duration:.0f} ns', 'amplitude (a.u.)', 'P(1)')
    plot_fit(ax, amps, p1, fit, hollow=True, data_color='tab:blue', fit_color='tab:blue')
    return Result('rabi_amp', {'amp': amps, 'p1': p1}, fit, meta, fig)

def gmm_readout(env: Env, k: int, shots: int=2000, update_cal: bool=False) -> Result:
    iq0 = env.measure([], shots, [k], name='gmm_0')[k]
    iq1 = env.measure([Segment([env.x_pulse(k)])], shots, [k], name='gmm_1')[k]
    fit = F.fit_two_gaussians_iq(iq0, iq1)
    if update_cal and fit['status'] == 'finished' and (fit['params']['snr'] > 1.5):
        env.cal.iq_angle[k] = fit['params']['angle']
        env.cal.iq_threshold[k] = fit['params']['threshold']
    fig = iq_figure(f'Readout IQ q{k}', [iq0, iq1], fit)
    return Result('gmm_readout', {'iq0': iq0, 'iq1': iq1}, fit, {'qubit': k, 'shots': shots}, fig)

def ramsey(env: Env, k: int, delays: np.ndarray, virtual_detuning: float=0.002, shots: int=200, spectator: Optional[dict]=None) -> Result:
    prep = []
    if spectator:
        prep = [Segment([env.x_pulse(j) for j, lvl in spectator.items() if lvl == 1])]
    p1 = []
    sequences = []
    for tau in delays:
        phi = 2 * np.pi * virtual_detuning * tau
        seq = prep + [Segment([env.x_pulse(k, np.pi / 2)]), float(tau), Segment([env.x_pulse(k, np.pi / 2, phase=phi)])]
        sequences.append(seq)
    for result in env.measure_many(sequences, shots, [k], name='ramsey'):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    fit = F.fit_damped_cosine(delays, p1)
    meta = {'qubit': k, 'virtual_detuning_GHz': virtual_detuning, 'shots': shots, 'spectator': spectator}
    title = f'Ramsey q{k}' + (f' spectator={spectator}' if spectator else '')
    fig = ramsey_figure(title, delays, [p1], [fit])
    return Result('ramsey', {'delay_ns': delays, 'p1': p1}, fit, meta, fig)

def t1(env: Env, k: int, delays: np.ndarray, shots: int=200) -> Result:
    p1 = []
    sequences = []
    for tau in delays:
        seq = [Segment([env.x_pulse(k)]), float(tau)]
        sequences.append(seq)
    for result in env.measure_many(sequences, shots, [k], name='t1'):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    fit = F.fit_exp_decay(delays, p1)
    meta = {'qubit': k, 'shots': shots, 'T1_ns': fit['params'].get('tau')}
    fig, ax = new_fig(f'T1 q{k}', 'delay (ns)', 'P(1)')
    plot_fit(ax, delays, p1, fit)
    return Result('t1', {'delay_ns': delays, 'p1': p1}, fit, meta, fig)

def echo(env: Env, k: int, delays: np.ndarray, shots: int=200) -> Result:
    p1 = []
    sequences = []
    for tau in delays:
        seq = [Segment([env.x_pulse(k, np.pi / 2)]), float(tau / 2), Segment([env.y_pulse(k)]), float(tau / 2), Segment([env.x_pulse(k, np.pi / 2)])]
        sequences.append(seq)
    for result in env.measure_many(sequences, shots, [k], name='echo'):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    fit = F.fit_exp_decay(delays, p1)
    meta = {'qubit': k, 'shots': shots, 'T2echo_ns': fit['params'].get('tau')}
    fig, ax = new_fig(f'Hahn echo q{k}', 'total delay (ns)', 'P(1)')
    plot_fit(ax, delays, p1, fit)
    return Result('echo', {'delay_ns': delays, 'p1': p1}, fit, meta, fig)

def pingpong(env: Env, k: int, n_list: List[int], shots: int=200, amp: Optional[float]=None) -> Result:
    p1 = []
    sequences = []
    for N in n_list:
        seq = [Segment([env.x_pulse(k, np.pi / 2, amp=None if amp is None else amp / 2)])]
        seq += [Segment([env.x_pulse(k, amp=amp)]) for _ in range(N)]
        sequences.append(seq)
    for result in env.measure_many(sequences, shots, [k], name='pingpong'):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    n = np.array(n_list)
    y = 2 * (p1 - 0.5) * (-1.0) ** n
    fit = F.fit_sine_error(n, y)
    eps = fit['params']['eps']
    cur = env.cal.pi_amp.get(k) if amp is None else amp
    meta = {'qubit': k, 'shots': shots}
    fig, ax = new_fig(f'PingPong q{k}: eps={eps * 1000.0:.2f} mrad/pulse', 'Pulse count (X180)', 'Parity-corrected 2P(1) − 1')
    plot_fit(ax, n, y, fit)
    return Result('pingpong', {'n': n, 'p1': p1, 'signed_error': y}, fit, meta, fig)

def repeated_gate(env: Env, k: int, n_list: List[int], gate: str='x90', shots: int=200) -> Result:
    p1 = []
    sequences = []
    for N in n_list:
        seq = [Segment([env.x_pulse(k, np.pi / 2)]) for _ in range(4 * N + 1)]
        sequences.append(seq)
    for result in env.measure_many(sequences, shots, [k], name='repeated_x90'):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    n = np.array(n_list)
    y = 2 * (p1 - 0.5)
    fit = F.fit_sine_error(4 * n, y)
    eps = fit['params']['eps']
    cur = env.cal.pi_amp.get(k)
    meta = {'qubit': k, 'shots': shots}
    fig, ax = new_fig(f'Repeated X90 q{k}: eps={eps * 1000.0:.2f} mrad', 'Amplification count (4N)', '2P(1) − 1')
    plot_fit(ax, 4 * n, y, fit)
    return Result('repeated_x90', {'n': n, 'p1': p1}, fit, meta, fig)

def drag_calibration(env: Env, k: int, betas: np.ndarray, shots: int=400, reps: int=1, include_reference: bool=True) -> Result:
    dA, dB, leak = ([], [], [])
    for b in betas:
        A = [Segment([env.x_pulse(k, np.pi / 2, drag=float(b))]), Segment([env.y_pulse(k, drag=float(b))])] * reps
        B = [Segment([env.y_pulse(k, np.pi / 2, drag=float(b))]), Segment([env.x_pulse(k, drag=float(b))])] * reps
        dA.append(env.p1(env.measure(A, shots, [k], name='drag_A')[k], k))
        dB.append(env.p1(env.measure(B, shots, [k], name='drag_B')[k], k))
        if include_reference:
            pops = env.sim.run([Segment([env.x_pulse(k, drag=float(b))]) for _ in range(10)])
            leak.append(env.sim.marginal(pops, k)[2:].sum())
    dA, dB, leak = map(np.array, (dA, dB, leak))
    diff = dA - dB
    fit = F.fit_linear(betas, diff)
    meta = {'qubit': k, 'shots': shots}
    if include_reference:
        meta.update(_truth_leakage_after_10_x180=leak.tolist(), _truth_beta_min_leakage=float(betas[np.argmin(leak)]))
    import matplotlib.pyplot as plt
    fig, (ax, diff_ax) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    fig.suptitle(f'DRAG q{k}', fontsize=11)
    ax.plot(betas, dA, 'o', ms=4, color='blue', alpha=0.6, label='X90·Y180')
    ax.plot(betas, dB, 's', ms=4, color='red', alpha=0.6, label='Y90·X180')
    ax.set(title='Measured sequences', xlabel='DRAG coefficient beta', ylabel='P(1)')
    diff_ax.set(title='Sequence difference', xlabel='DRAG coefficient beta', ylabel='P(1) A − P(1) B')
    plot_fit(diff_ax, betas, diff, fit)
    diff_ax.axhline(0, color='gray', ls='--', lw=1)
    for panel in (ax, diff_ax):
        panel.grid(alpha=0.25)
        panel.legend(fontsize=8)
    if include_reference:
        ax2 = ax.twinx()
        ax2.plot(betas, leak, 'k--', lw=1, label='true leakage (10×X180)')
        ax2.set_ylabel('leakage')
        ax2.legend(fontsize=8, loc='upper right')
    return Result('drag', {'beta': betas, 'p1_A': dA, 'p1_B': dB, 'diff': diff}, fit, meta, fig)
