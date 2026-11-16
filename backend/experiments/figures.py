from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .common import plot_fit

def ramsey_figure(title, delays, populations, fits, labels=None):
    x = np.asarray(delays, float)
    labels = labels or [''] * len(populations)
    fig, axes = plt.subplots(len(populations), 2, squeeze=False, figsize=(11, 3.8 * len(populations)), layout='constrained')
    fig.suptitle(title, fontsize=11)
    order = np.argsort(x)
    xs = x[order]
    dx = np.diff(xs)
    uniform = len(x) >= 4 and np.all(np.isfinite(x)) and np.all(dx > 0) and np.allclose(dx, dx[0], rtol=1e-05, atol=1e-10)
    for (spectrum, trace), values, fit, label in zip(axes, populations, fits, labels):
        y = np.asarray(values)
        spectrum.set(title=f'Frequency spectrum {label}', xlabel='Frequency (MHz)', ylabel='FFT magnitude')
        if uniform and np.all(np.isfinite(y)):
            freq = np.fft.rfftfreq(len(x), d=dx[0]) * 1000
            spectrum.plot(freq, np.abs(np.fft.rfft(y[order] - np.mean(y))), color='blue', lw=1)
            f = fit.get('params', {}).get('freq')
            if fit.get('status') == 'finished' and f is not None and np.isfinite(f):
                spectrum.axvline(f * 1000, color='red', ls='--', lw=1, label=f'Fit: {f * 1000:.4g} MHz')
                spectrum.legend(fontsize=8)
        else:
            spectrum.text(0.5, 0.5, 'FFT unavailable\nRequires ≥4 uniformly spaced, finite samples', transform=spectrum.transAxes, ha='center', va='center', fontsize=9)
        plot_fit(trace, x, y, fit, annotate=False)
        p = fit.get('params', {})
        if fit.get('status') == 'finished' and all((k in p and np.isfinite(p[k]) for k in ('tau', 'amp', 'offset'))) and (p['tau'] > 0):
            xx = np.linspace(xs[0], xs[-1], 800)
            for sign in (-1, 1):
                trace.plot(xx, p['offset'] + sign * abs(p['amp']) * np.exp(-xx / p['tau']), '--', color='#66bb66', lw=1, label='Decay envelope' if sign == 1 else None)
        for line in trace.lines:
            line.set_xdata(np.asarray(line.get_xdata()) / 1000)
        trace.relim()
        trace.autoscale_view()
        info = f"Fit: {fit.get('status', 'failed')}"
        if fit.get('status') == 'finished':
            if 'freq' in p:
                info += f"; f={p['freq'] * 1000:.4g} MHz"
            if 'tau' in p:
                info += f"; τ={p['tau'] / 1000:.4g} μs"
            if np.isfinite(fit.get('r2', np.nan)):
                info += f"; R²={fit['r2']:.3f}"
        trace.set(title=f'Time domain {label}\n{info}', xlabel='Evolution time (μs)', ylabel='P(1)')
        trace.legend(fontsize=8)
        for ax in (spectrum, trace):
            ax.grid(alpha=0.25)
            ax.title.set_fontsize(9)
    return fig

def iq_figure(title, clouds, fit):
    clouds = [np.asarray(c) for c in clouds]
    p = fit.get('params', {})
    means = p.get('means')
    if means is None and 'mean0_I' in p:
        means = np.array([p[f'mean{k}_I'] + 1j * p[f'mean{k}_Q'] for k in range(2)])
    means = None if means is None else np.asarray(means)
    valid = fit.get('status') == 'finished' and means is not None and np.all(np.isfinite(means))
    fig, axes = plt.subplots(1, len(clouds), squeeze=False, sharex=True, sharey=True, figsize=(4 * len(clouds), 4.5), layout='constrained')
    fidelity = p.get('fidelity')
    fig.suptitle(title + (f' — fitted assignment F={fidelity:.3f}' if fidelity is not None else ''), fontsize=11)
    colors = ['tab:blue', 'red', 'tab:green']
    for j, (ax, iq) in enumerate(zip(axes[0], clouds)):
        if valid:
            if len(means) == 2 and 'angle' in p and ('threshold' in p):
                assigned = (np.real(iq * np.exp(-1j * p['angle'])) > p['threshold']).astype(int)
            else:
                assigned = np.argmin(abs(iq[:, None] - means[None, :]), axis=1)
            for k, mean in enumerate(means):
                selected = iq[assigned == k]
                color = colors[k % len(colors)]
                ax.scatter(selected.real, selected.imag, s=4, alpha=0.55, color=color, label=f'{k}: {np.mean(assigned == k) * 100:.2f}%', rasterized=True)
                ax.plot(mean.real, mean.imag, 'x', color=color, ms=8)
                sigma = p.get('sigma', np.nan)
                if np.isfinite(sigma) and sigma > 0:
                    ax.add_patch(Circle((mean.real, mean.imag), 2 * sigma, fill=False, color=color, lw=1.3))
            ax.legend(fontsize=8, loc='upper right', markerscale=1.5)
        else:
            ax.scatter(iq.real, iq.imag, s=4, alpha=0.55, color='tab:blue', rasterized=True)
            ax.text(0.02, 0.98, 'Fit unavailable', transform=ax.transAxes, va='top')
        ax.set(title=f'Distribution {j}', xlabel='I', ylabel='Q')
        ax.set_aspect('equal', adjustable='box')
    return fig

def power_figure(title, freqs, amps, iq):
    x, y, z = (np.asarray(freqs) * 1000, np.asarray(amps), np.asarray(iq))
    xi, yi = (np.argsort(x), np.argsort(y))
    x, y, z = (x[xi], y[yi], z[np.ix_(yi, xi)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    fig.suptitle(title, fontsize=11)
    for ax, values, cmap, label in zip(axes, (abs(z), np.angle(z)), ('viridis', 'RdBu'), ('Magnitude', 'Phase (rad)')):
        if len(x) > 1 and len(y) > 1 and np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0):
            im = ax.pcolormesh(x, y, values, shading='nearest', cmap=cmap, rasterized=True)
        else:
            xx, yy = np.meshgrid(x, y)
            im = ax.scatter(xx.ravel(), yy.ravel(), c=values.ravel(), cmap=cmap, marker='s', s=45)
        ax.set(title=label + ' sweep', xlabel='Drive frequency (MHz)', ylabel='Drive amplitude (a.u.)')
        ax.ticklabel_format(axis='x', useOffset=False)
        fig.colorbar(im, ax=ax, label=label, pad=0.02)
    return fig
