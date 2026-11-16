from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit

def _r2(y, yfit):
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-30
    return float(1 - ss_res / ss_tot)

def _fit(model: Callable, x, y, p0, names, bounds=(-np.inf, np.inf), min_r2=0.5, maxfev=20000):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        perr = np.sqrt(np.abs(np.diag(pcov)))
    except Exception as e:
        return {'status': 'failed', 'error': f'curve_fit failed: {e}', 'params': {}, 'errors': {}, 'r2': float('nan')}
    yfit = model(x, *popt)
    r2 = _r2(y, yfit)
    return {'status': 'finished', 'params': dict(zip(names, map(float, popt))), 'errors': dict(zip(names, map(float, perr))), 'r2': r2, 'yfit': yfit}

def lorentzian(x, x0, gamma, a, c):
    return c + a / (1 + ((x - x0) / (gamma / 2)) ** 2)

def cosine(x, f, phi, a, c):
    return c + a * np.cos(2 * np.pi * f * x + phi)

def damped_cosine(x, f, phi, a, c, tau):
    return c + a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi)

def exp_decay(x, tau, a, c):
    return c + a * np.exp(-x / tau)

def gauss_decay(x, tau, a, c):
    return c + a * np.exp(-(x / tau) ** 2)

def _guess_freq(x, y):
    x = np.asarray(x)
    y = np.asarray(y) - np.mean(y)
    dx = np.mean(np.diff(x))
    n = len(x)
    freqs = np.fft.rfftfreq(n, dx)
    spec = np.abs(np.fft.rfft(y))
    spec[0] = 0
    return float(freqs[np.argmax(spec)]) if n > 3 else 1.0 / (x[-1] - x[0] + 1e-12)

def fit_lorentzian(x, y, dip: Optional[bool]=None):
    x = np.asarray(x)
    y = np.asarray(y)
    if dip is None:
        dip = np.median(y) - y.min() > y.max() - np.median(y)
    c0 = np.median(y)
    x0 = x[np.argmin(y)] if dip else x[np.argmax(y)]
    a0 = y.min() - c0 if dip else y.max() - c0
    span = x.max() - x.min()
    half = c0 + a0 / 2
    mask = y < half if dip else y > half
    g0 = max(x[mask].max() - x[mask].min() if mask.sum() > 1 else span / 20, span / 200)
    res = _fit(lorentzian, x, y, [x0, g0, a0, c0], ['x0', 'fwhm', 'amp', 'offset'], bounds=([x.min(), span / 1000, -np.inf, -np.inf], [x.max(), span, np.inf, np.inf]))
    res['model'] = 'lorentzian'
    return res

def _grid_guesses(x, y, f0: Optional[float]=None, n_grid: int=200, n_keep: int=3):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = np.min(np.diff(np.sort(x))) if len(x) > 1 else 1.0
    span = x.max() - x.min() + 1e-12
    grid = np.linspace(0.25 / span, 0.5 / dx, n_grid)
    if f0 is not None:
        grid = np.append(grid, f0)
    out = []
    for fg in grid:
        A = np.vstack([np.ones_like(x), np.cos(2 * np.pi * fg * x), np.sin(2 * np.pi * fg * x)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        c, a, b = sol
        out.append((np.sum((y - A @ sol) ** 2), fg, float(np.arctan2(-b, a)), float(np.hypot(a, b)), float(c)))
    out.sort(key=lambda t: t[0])
    keep = out[:n_keep]
    if f0 is not None:
        keep.append(out[[o[1] for o in out].index(f0)]) if f0 in [o[1] for o in out] else None
    return [(fg, ph, a, c) for _, fg, ph, a, c in keep]

def fit_cosine(x, y, f0: Optional[float]=None):
    x = np.asarray(x)
    y = np.asarray(y)
    best = None
    for fg, ph, a, c in _grid_guesses(x, y, f0):
        r = _fit(cosine, x, y, [fg, ph, max(a, 0.001), c], ['freq', 'phase', 'amp', 'offset'], bounds=([0, -2 * np.pi, 0, -np.inf], [np.inf, 2 * np.pi, 1.5, np.inf]))
        if best is None or r['r2'] > best['r2']:
            best = r
    best['model'] = 'cosine'
    return best

def fit_damped_cosine(x, y, f0: Optional[float]=None):
    x = np.asarray(x)
    y = np.asarray(y)
    tau0 = x.max() - x.min()
    best = None
    for fg, ph, a, c in _grid_guesses(x, y, f0):
        r = _fit(damped_cosine, x, y, [fg, ph, max(a, 0.001), c, tau0], ['freq', 'phase', 'amp', 'offset', 'tau'], bounds=([0, -2 * np.pi, 0, -np.inf, tau0 / 100], [np.inf, 2 * np.pi, 1.5, np.inf, tau0 * 100]))
        if best is None or r['r2'] > best['r2']:
            best = r
    best['model'] = 'damped_cosine'
    return best

def fit_exp_decay(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    c0 = y[-1]
    a0 = y[0] - c0
    tau0 = (x.max() - x.min()) / 3
    r = _fit(exp_decay, x, y, [tau0, a0, c0], ['tau', 'amp', 'offset'], bounds=([tau0 / 1000, -np.inf, -np.inf], [tau0 * 1000, np.inf, np.inf]))
    r['model'] = 'exp_decay'
    return r

def sine_error(n, eps, phi0, a):
    return a * np.sin(n * eps + phi0)

def fit_sine_error(n, y):
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    lin = fit_linear(n, y)
    best = None
    for eps0 in [lin['params']['slope'], 0.05, -0.05, 0.2, -0.2]:
        r = _fit(sine_error, n, y, [eps0, 0.0, 1.0], ['eps', 'phi0', 'amp'], bounds=([-np.pi / 2, -np.pi, 0.2], [np.pi / 2, np.pi, 1.2]), min_r2=-np.inf)
        if best is None or r['r2'] > best['r2']:
            best = r
    best['status'] = 'finished'
    best['model'] = 'sine_error'
    return best

def fit_linear(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    A = np.vstack([x, np.ones_like(x)]).T
    sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yfit = A @ sol
    return {'status': 'finished', 'params': {'slope': float(sol[0]), 'intercept': float(sol[1])}, 'errors': {}, 'r2': _r2(y, yfit), 'yfit': yfit, 'model': 'linear'}

def fit_two_gaussians_iq(iq0: np.ndarray, iq1: np.ndarray, n_iter: int=20):
    z = np.concatenate([iq0, iq1])
    X = np.stack([z.real, z.imag], axis=1)
    mu = np.array([[iq0.real.mean(), iq0.imag.mean()], [iq1.real.mean(), iq1.imag.mean()]])
    sig = np.std(X, axis=0).mean()
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        logp = np.log(w)[None, :] - d2 / (2 * sig ** 2)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
        mu = np.einsum('nk,nd->kd', r, X, optimize=False) / r.sum(0)[:, None]
        sig = np.sqrt((r * d2).sum() / (2 * r.sum()))
    m0 = mu[0, 0] + 1j * mu[0, 1]
    m1 = mu[1, 0] + 1j * mu[1, 1]
    angle = float(np.angle(m1 - m0))
    thr = float(((m0 + m1) / 2 * np.exp(-1j * angle)).real)
    a0 = ((iq0 * np.exp(-1j * angle)).real > thr).mean()
    a1 = ((iq1 * np.exp(-1j * angle)).real > thr).mean()
    fid = 1 - (a0 + (1 - a1)) / 2
    sep = abs(m1 - m0) / sig
    return {'status': 'finished', 'model': 'gmm2', 'r2': float('nan'), 'params': {'mean0_I': float(m0.real), 'mean0_Q': float(m0.imag), 'mean1_I': float(m1.real), 'mean1_Q': float(m1.imag), 'sigma': float(sig), 'angle': angle, 'threshold': thr, 'fidelity': float(fid), 'p10': float(a0), 'p01': float(1 - a1), 'snr': float(sep)}, 'errors': {}}

def fit_gaussians_iq(clouds, n_iter: int=30):
    K = len(clouds)
    z = np.concatenate(clouds)
    X = np.stack([z.real, z.imag], axis=1)
    mu = np.array([[c.real.mean(), c.imag.mean()] for c in clouds])
    sig = np.mean([np.std(np.stack([c.real, c.imag], 1), axis=0).mean() for c in clouds])
    w = np.ones(K) / K
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        logp = np.log(w)[None, :] - d2 / (2 * sig ** 2)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
        mu = np.einsum('nk,nd->kd', r, X, optimize=False) / r.sum(0)[:, None]
        sig = np.sqrt((r * d2).sum() / (2 * r.sum()))
    means = mu[:, 0] + 1j * mu[:, 1]
    conf = np.zeros((K, K))
    for i, c in enumerate(clouds):
        lab = np.argmin(np.abs(c[:, None] - means[None, :]), axis=1)
        for j in range(K):
            conf[i, j] = np.mean(lab == j)
    sep = min((abs(means[i] - means[j]) for i in range(K) for j in range(i + 1, K))) / sig
    return {'status': 'finished', 'model': f'gmm{K}', 'r2': float('nan'), 'params': {'means': means, 'sigma': float(sig), 'confusion': conf, 'fidelity': float(np.trace(conf) / K), 'snr_min': float(sep)}, 'errors': {}}
