from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass
class Result:
    name: str
    points: Dict[str, Any]
    fit: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    figure: Optional[plt.Figure] = None

    def save(self, path: str):
        if self.figure is not None:
            self.figure.savefig(path, dpi=110, bbox_inches='tight')
            plt.close(self.figure)
        return path

    def summary(self) -> str:
        p = {k: round(v, 6) if isinstance(v, float) else v for k, v in self.fit.get('params', {}).items()}
        status = self.fit.get('status', 'failed')
        return f"[{self.name}] fit {status} r2={self.fit.get('r2', float('nan')):.3f} params={p}"

def new_fig(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return (fig, ax)

def fitted_curve(x, fit):
    from .. import fitting as F
    p = fit.get('params', {})
    models = {'lorentzian': (F.lorentzian, ('x0', 'fwhm', 'amp', 'offset')), 'cosine': (F.cosine, ('freq', 'phase', 'amp', 'offset')), 'damped_cosine': (F.damped_cosine, ('freq', 'phase', 'amp', 'offset', 'tau')), 'exp_decay': (F.exp_decay, ('tau', 'amp', 'offset')), 'sine_error': (F.sine_error, ('eps', 'phi0', 'amp')), 'linear': (lambda t, a, b: a * t + b, ('slope', 'intercept'))}
    entry = models.get(fit.get('model'))
    if fit.get('status') != 'finished' or entry is None:
        return None
    model, names = entry
    if not all((k in p and p[k] is not None and np.isfinite(p[k]) for k in names)):
        return None
    with np.errstate(all='ignore'):
        return model(np.asarray(x), *(p[k] for k in names))

def plot_fit(ax, x, y, fit, xfine=None, label_data='Data', *, hollow=False, data_color='blue', fit_color='red', annotate=True, data_line=False):
    x = np.asarray(x)
    ax.plot(x, y, 'o-' if data_line else 'o', ms=4, lw=1, alpha=0.6, color=data_color, mfc='none' if hollow else data_color, label=label_data)
    if fit.get('status') == 'finished' and 'yfit' in fit:
        xx = np.linspace(np.min(x), np.max(x), 800) if xfine is None else xfine
        yy = fitted_curve(xx, fit)
        if yy is None:
            order = np.argsort(x)
            xx, yy = (x[order], np.asarray(fit['yfit'])[order])
        ax.plot(xx, yy, '-', color=fit_color, lw=1.5, label='Fit')
        p = fit.get('params', {})
        if fit.get('model') == 'lorentzian' and 'x0' in p:
            ax.axvline(p['x0'], color='green', ls='--', lw=1, label=f"Resonance: {p['x0']:.6g}")
    if annotate:
        p = fit.get('params', {})
        lines = [f"Fit: {fit.get('status', 'failed')}"]
        r2 = fit.get('r2', np.nan)
        if np.isfinite(r2):
            lines.append(f'R² = {r2:.3f}')
        for key in ('freq', 'tau', 'eps'):
            if key in p and p[key] is not None:
                if key == 'freq' and '(ns)' in ax.get_xlabel():
                    lines.append(f'f = {p[key] * 1000:.4g} MHz')
                elif key == 'tau' and '(ns)' in ax.get_xlabel():
                    lines.append(f'τ = {p[key] / 1000:.4g} μs')
                elif key == 'freq':
                    lines.append(f'f = {p[key]:.4g} cycles/a.u.')
                elif key == 'eps':
                    lines.append(f'ε = {p[key]:.4g} rad')
                else:
                    lines.append(f'{key} = {p[key]:.4g}')
        ax.set_title(ax.get_title() + '\n' + '; '.join(lines), fontsize=9)
    ax.legend(fontsize=8)
