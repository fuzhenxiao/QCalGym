import numpy as np
import matplotlib.pyplot as plt
from .core import Segment
from .experiments.common import Result
from .fitting import fit_cosine

def rabi_phase_scan(env, k, amps, phases, shots=512):
    grid = []
    fits = []
    for phase in phases:
        sequences = [[Segment([env.x_pulse(k, amp=float(a), phase=float(phase))])] for a in amps]
        y = np.array([env.p1(iq[k], k) for iq in env.measure_many(sequences, shots, [k], name='phase_rabi')])
        f = fit_cosine(np.asarray(amps), y)
        if f['status'] != 'finished':
            raise ValueError('Phase Rabi fit did not converge')
        grid.append(y)
        fits.append(f)
    rates = np.array([f['params']['freq'] for f in fits])
    phases = np.array(phases)
    design = np.column_stack([np.cos(phases) ** 2, np.sin(phases) ** 2, np.sin(2 * phases)])
    coeff = np.linalg.lstsq(design, rates ** 2, rcond=None)[0]
    gain = np.sqrt(coeff[1] / coeff[0])
    skew = np.arcsin(np.clip(coeff[2] / np.sqrt(coeff[0] * coeff[1]), -1, 1))
    predicted = design @ coeff
    relative = float(np.sqrt(np.mean((rates ** 2 - predicted) ** 2)) / np.mean(rates ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for phase, y, f in zip(phases, grid, fits):
        axes[0].plot(amps, y, 'o', ms=3, label=f'{phase / np.pi:.2g} pi')
        axes[0].plot(amps, f['yfit'], lw=1)
    x = np.linspace(min(phases), max(phases), 201)
    B = np.column_stack([np.cos(x) ** 2, np.sin(x) ** 2, np.sin(2 * x)])
    axes[1].plot(phases, rates / rates[0], 'o')
    axes[1].plot(x, np.sqrt(np.maximum(B @ coeff, 0)) / rates[0])
    axes[1].axhline(1, color='gray', ls='--')
    axes[0].set(xlabel='Drive amplitude', ylabel='P(1)', title=f'q{k} Rabi at different command phases')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Command phase (rad)', ylabel='Rabi rate / phase-0 rate', title='Phase dependence of drive gain')
    for ax in axes:
        ax.grid(alpha=0.25)
    return Result('rabi_phase_scan', dict(amp=amps, phase_rad=phases, p1=np.asarray(grid), rabi_rate=rates), dict(status='finished', model='quadrature_gain_ellipse', params=dict(gain_ratio=float(gain), skew_rad=float(skew), relative_fit_residual=relative, min_rabi_r2=min((f['r2'] for f in fits)), rates=rates.tolist()), individual_fits=fits), dict(shots=shots), fig)

def install(n):
    from .benchmark.catalog import register_experiment
    name = 'rabi_phase_scan'
    spec = dict(description='Amplitude Rabi at multiple command phases; fit phase-dependent effective drive gain from sampled populations.', parameters=dict(type='object', properties=dict(k=dict(type='integer', minimum=0, maximum=n - 1), amps=dict(type='array', items=dict(type='number', minimum=0, maximum=2), minItems=11, maxItems=41), phases=dict(type='array', items=dict(type='number', minimum=-7, maximum=7), minItems=5, maxItems=9), shots=dict(type='integer', minimum=16, maximum=4096, default=512)), required=['k', 'amps', 'phases'], additionalProperties=False))
    register_experiment(name, rabi_phase_scan, spec, lambda p: len(p['amps']) * len(p['phases']))
    return {name: spec}
