from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ..core import Pulse, Segment
from ..env import Env
from .. import fitting as F
from .common import Result, new_fig, plot_fit
from .single import qubit_spec
from .figures import iq_figure, ramsey_figure, power_figure
from ..acquisition_cache import reuse_acquisitions

def gmm3_readout(env: Env, k: int, shots: int=2000, update_cal: bool=True) -> Result:
    iq0 = env.measure([], shots, [k], name='gmm3_0')[k]
    iq1 = env.measure([Segment([env.x_pulse(k)])], shots, [k], name='gmm3_1')[k]
    iq2 = env.measure([Segment([env.x_pulse(k)]), Segment([env.x12_pulse(k)])], shots, [k], name='gmm3_2')[k]
    fit = F.fit_gaussians_iq([iq0, iq1, iq2])
    if update_cal and fit['status'] == 'finished' and (fit['params']['snr_min'] > 1.5):
        env.cal.iq_means3[k] = list(fit['params']['means'])
    fig = iq_figure(f'3-state readout q{k}', [iq0, iq1, iq2], fit)
    conf = fit['params']['confusion']
    return Result('gmm3_readout', {'iq0': iq0, 'iq1': iq1, 'iq2': iq2}, fit, {'qubit': k, 'shots': shots, 'confusion': conf.tolist(), 'p2_given_2': float(conf[2, 2])}, fig)

@reuse_acquisitions
def leakage_vs_pulses(env: Env, k: int, n_list: List[int], drag: Optional[float]=None, shots: int=1000, include_reference: bool=True) -> Result:
    p2, p2_true = ([], [])
    for N in n_list:
        seq = [Segment([env.x_pulse(k, drag=drag)]) for _ in range(N)]
        iq = env.measure(seq, shots, [k], name='leakage')[k]
        p2.append(env.p_levels(iq, k)[2])
        if include_reference:
            p2_true.append(env.sim.marginal(env._last_pops, k)[2:].sum())
    p2 = np.array(p2)
    p2_true = np.array(p2_true)
    n = np.array(n_list)
    fit = F.fit_linear(n, p2)
    meta = {'qubit': k, 'drag': drag, 'shots': shots}
    if include_reference:
        meta['_truth_p2'] = p2_true.tolist()
    fig, ax = new_fig(f"Leakage q{k}, DRAG beta={drag}: {fit['params']['slope'] * 1000.0:.2f}e-3 / X180", 'number of X180', 'P(2)')
    ax.plot(n, p2, 'o', ms=3.5, label='3-state readout')
    if include_reference:
        ax.plot(n, p2_true, 'k--', lw=1, label='true P(2)')
    ax.legend(fontsize=8)
    return Result('leakage_vs_pulses', {'n': n, 'p2': p2}, fit, meta, fig)

def spec_vs_power(env: Env, k: int, freqs: np.ndarray, amps: List[float], duration: float=2000.0, shots: int=150, shape: str='gaussian') -> Result:
    traces, fits, iq_rows = ({}, {}, [])
    for a in amps:
        r = qubit_spec(env, k, freqs, amp=a, duration=duration, shots=shots, shape=shape)
        plt.close(r.figure)
        traces[a] = r.points['response']
        fits[a] = r.fit
        iq_rows.append(r.points['iq_mean'])
    iq_grid = np.asarray(iq_rows)
    fig = power_figure(f'Spectroscopy vs power q{k}', freqs, amps, iq_grid)
    fit = {'status': 'finished', 'model': 'per_power_lorentzian', 'r2': float('nan'), 'params': {f'fwhm_MHz_amp{a}': fits[a]['params'].get('fwhm', np.nan) * 1000.0 for a in amps}, 'errors': {}}
    return Result('spec_vs_power', {'freq_GHz': freqs, 'amp': np.asarray(amps), 'iq_mean': iq_grid, **{f'resp_amp{a}': traces[a] for a in amps}}, fit, {'qubit': k, 'amps': amps, 'duration': duration}, fig)

def _axis_pulse(env: Env, k: int, phi: float) -> Pulse:
    return env.x_pulse(k, np.pi / 2, phase=np.pi / 2 - phi)

@reuse_acquisitions
def xy_plane_cnot_calibration(env: Env, control: int, target: int, cr_amp: float, cr_duration: float, angles: np.ndarray, cr_phase: Optional[float]=None, shots: int=300, rise: Optional[float]=None) -> Result:
    data = {}
    for ctrl in (0, 1):
        vals = []
        for phi in angles:
            seq = [Segment([env.x_pulse(control)])] if ctrl else []
            seq += [env.cr_segment(control, target, amp=cr_amp, duration=cr_duration, phase=cr_phase, rise=rise), Segment([_axis_pulse(env, target, float(phi))])]
            iq = env.measure(seq, shots, [target], name='xy_cnot')[target]
            vals.append(1 - 2 * env.p1(iq, target))
        data[ctrl] = np.array(vals)
    f0 = F.fit_cosine(angles, data[0], f0=1 / (2 * np.pi))
    f1 = F.fit_cosine(angles, data[1], f0=1 / (2 * np.pi))
    phi0 = -f0['params'].get('phase', np.nan)
    phi1 = -f1['params'].get('phase', np.nan)
    axis = float(np.angle(np.exp(2j * (phi0 + np.pi / 2)) + np.exp(2j * (phi1 + np.pi / 2))) / 2)
    fit = {'status': 'finished' if f0['status'] == f1['status'] == 'finished' else 'failed', 'model': 'two_cosines_vs_angle', 'r2': min(f0['r2'], f1['r2']), 'params': {'bloch_angle_ctrl0': float(phi0 % (2 * np.pi)), 'bloch_angle_ctrl1': float(phi1 % (2 * np.pi)), 'zx_axis_angle': float(axis), 'amp_ctrl0': f0['params'].get('amp'), 'amp_ctrl1': f1['params'].get('amp')}, 'errors': {}}
    fig, ax = new_fig(f'XY-plane CNOT calibration c{control}->t{target}: axis={axis:.2f} rad', 'measurement axis angle (rad)', '<n_phi> of target')
    ax.plot(angles, data[0], 'o', ms=3, label='control |0>')
    ax.plot(angles, data[1], 's', ms=3, label='control |1>')
    if f0['status'] == 'finished':
        ax.plot(angles, f0['yfit'], '-', lw=1)
    if f1['status'] == 'finished':
        ax.plot(angles, f1['yfit'], '-', lw=1)
    ax.legend(fontsize=8)
    return Result('xy_plane_cnot_calibration', {'angle': angles, 'n_ctrl0': data[0], 'n_ctrl1': data[1]}, fit, {'control': control, 'target': target, 'cr_amp': cr_amp, 'cr_duration': cr_duration, 'shots': shots}, fig)

def ramsey_with_tones(env: Env, k: int, delays: np.ndarray, tones: List[Tuple[int, float, float, float]], virtual_detuning: float=0.002, shots: int=200, spectator: Optional[dict]=None, name: str='ramsey_tones', rise: float=60.0, *, _make_figure: bool=True) -> Result:
    prep = [Segment([env.x_pulse(j) for j, lvl in (spectator or {}).items() if lvl == 1])] if spectator else []
    p1 = []
    sequences = []
    for tau in delays:
        phi = 2 * np.pi * virtual_detuning * tau
        seq = prep + [Segment([env.x_pulse(k, np.pi / 2)])]
        if tau > 0:
            seq += [Segment([env.drive_pulse(line, f, amp, float(tau) + 2 * rise, phase=ph, shape='gaussian_square', rise=rise) for line, f, amp, ph in tones])]
        seq += [Segment([env.x_pulse(k, np.pi / 2, phase=phi)])]
        sequences.append(seq)
    for result in env.measure_many(sequences, shots, [k], name=name):
        p1.append(env.p1(result[k], k))
    p1 = np.array(p1)
    fit = F.fit_damped_cosine(delays, p1)
    meta = {'qubit': k, 'tones': tones, 'virtual_detuning_GHz': virtual_detuning, 'spectator': spectator}
    fig = ramsey_figure(f'Ramsey q{k} with tones on', delays, [p1], [fit]) if _make_figure else None
    return Result(name, {'delay_ns': delays, 'p1': p1}, fit, meta, fig)

def stark_shift_scan(env: Env, k: int, tone_freq: float, amps: List[float], delays: np.ndarray, virtual_detuning: float=0.003, shots: int=200, include_reference: bool=True) -> Result:
    shifts, qs, tl = ([], [], [])
    from ..three_qubit_runtime import supported, ramsey_traces
    fits = ramsey_traces(env, k, delays, [([(k, tone_freq, a, 0.0)], None, 'stark') for a in amps], virtual_detuning, shots) if supported(env.dev) else None
    f_ref = env.sim.dressed_frequency(k) if include_reference else None
    for index, a in enumerate(amps):
        if fits is None:
            r = ramsey_with_tones(env, k, delays, [(k, tone_freq, a, 0.0)], virtual_detuning, shots, name='stark', _make_figure=env.dev.n > 2)
            if r.figure is not None:
                plt.close(r.figure)
            fit = r.fit
        else:
            fit = fits[index]
        det = fit['params'].get('freq', np.nan) - virtual_detuning
        shifts.append(det)
        if include_reference:
            qs.append(env.sim.driven_frequency(k, [(k, tone_freq, a, 0.0)]) - f_ref)
            Om = 2 * np.pi * env.dev.qubits[k].drive_scale * a
            dl = 2 * np.pi * (f_ref - tone_freq)
            tl.append(Om ** 2 / (2 * dl) / (2 * np.pi))
    shifts = -(np.array(shifts) - shifts[0]) if len(shifts) else np.array([])
    qs = np.array(qs)
    tl = np.array(tl)
    fit = {'status': 'finished', 'model': 'stark', 'r2': float('nan'), 'params': {'measured_shift_MHz': (shifts * 1000.0).tolist()}, 'errors': {}}
    if include_reference:
        fit['params'].update(quasistatic_MHz=(qs * 1000.0).tolist(), two_level_formula_MHz=(tl * 1000.0).tolist())
    fig, ax = new_fig(f'AC-Stark shift q{k}, tone at {tone_freq:.3f} GHz', 'tone amplitude', 'frequency shift (MHz)')
    ax.plot(amps, shifts * 1000.0, 'o', ms=4, label='Ramsey (simulated)')
    if include_reference:
        ax.plot(amps, qs * 1000.0, '-', lw=1.2, label='quasi-static spectrum (3-level)')
        ax.plot(amps, tl * 1000.0, '--', lw=1, label='two-level Ω²/2Δ')
    ax.legend(fontsize=8)
    return Result('stark_shift', {'amp': np.array(amps), 'shift_GHz': shifts}, fit, {'qubit': k, 'tone_freq': tone_freq}, fig)

def sizzle_zz_scan(env: Env, i: int, j: int, tone_freq: float, amps: List[float], delays: np.ndarray, phase_j: float=0.0, amp_ratio: float=1.0, virtual_detuning: float=0.003, shots: int=200, include_reference: bool=True) -> Result:
    zz_meas, zz_qs = ([], [])
    from ..three_qubit_runtime import supported, ramsey_traces
    branches = []
    if supported(env.dev):
        for a in amps:
            tones = [(i, tone_freq, a, 0.0), (j, tone_freq, a * amp_ratio, phase_j)]
            branches.extend([(tones, None, 'sizzle0'), (tones, {j: 1}, 'sizzle1')])
    fits = ramsey_traces(env, i, delays, branches, virtual_detuning, shots) if branches else None
    for index, a in enumerate(amps):
        tones = [(i, tone_freq, a, 0.0), (j, tone_freq, a * amp_ratio, phase_j)]
        if fits is None:
            r0 = ramsey_with_tones(env, i, delays, tones, virtual_detuning, shots, spectator=None, name='sizzle0', _make_figure=env.dev.n > 2)
            r1 = ramsey_with_tones(env, i, delays, tones, virtual_detuning, shots, spectator={j: 1}, name='sizzle1', _make_figure=env.dev.n > 2)
            if r0.figure is not None:
                plt.close(r0.figure)
            if r1.figure is not None:
                plt.close(r1.figure)
            f0, f1 = (r0.fit, r1.fit)
        else:
            f0, f1 = fits[2 * index:2 * index + 2]
        d0 = f0['params'].get('freq', np.nan) - virtual_detuning
        d1 = f1['params'].get('freq', np.nan) - virtual_detuning
        zz_meas.append(-(d1 - d0))
        if include_reference:
            zz_qs.append(env.sim.driven_zz(i, j, tones))
    zz_meas = np.array(zz_meas)
    zz_qs = np.array(zz_qs)
    fit = {'status': 'finished', 'model': 'sizzle', 'r2': float('nan'), 'params': {'zz_measured_kHz': (zz_meas * 1000000.0).tolist()}, 'errors': {}}
    if include_reference:
        fit['params'].update(zz_quasistatic_kHz=(zz_qs * 1000000.0).tolist(), zz_static_kHz=env.sim.zz(i, j) * 1000000.0)
    fig, ax = new_fig(f'siZZle q{i}-q{j}: tone {tone_freq:.3f} GHz, phase {phase_j:.2f}', 'tone amplitude', 'ZZ (kHz)')
    ax.plot(amps, zz_meas * 1000000.0, 'o', ms=4, label='conditional Ramsey (simulated)')
    if include_reference:
        ax.plot(amps, zz_qs * 1000000.0, '-', lw=1.2, label='quasi-static spectrum')
    ax.axhline(0, color='k', lw=0.6)
    ax.legend(fontsize=8)
    return Result('sizzle_zz', {'amp': np.array(amps), 'zz_GHz': zz_meas}, fit, {'i': i, 'j': j, 'tone_freq': tone_freq, 'phase_j': phase_j}, fig)

def calibrated_sizzle_ramsey(env: Env, i: int, j: int, delays: np.ndarray, **kw) -> Result:
    kw.setdefault('rise', env.cal.sizzle_rise.get((i, j), 60.0))
    return ramsey_with_tones(env, i, delays, env.sizzle_tones(i, j), **kw)
