from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import math
import cmath
import numpy as np
import qutip as qt
import scipy.linalg as la
from . import solver_fallback
from .numerics import expm, apply_propagator
from .device import DeviceConfig
TWO_PI = 2 * np.pi

@dataclass
class Pulse:
    line: int
    freq: float
    amp: float
    duration: float
    phase: float = 0.0
    shape: str = 'gaussian'
    drag: float = 0.0
    start: float = 0.0
    rise: float = 10.0
    n_sigma: float = 4.0
    drag_anharm: float = -0.33
    source_weights: Optional[list] = None
    smooth_edges: bool = False

    def envelope(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        e = np.array([self.env_scalar(x)[0] for x in t.ravel()]).reshape(t.shape)
        de = np.array([self.env_scalar(x)[1] for x in t.ravel()]).reshape(t.shape)
        return (e, de)

    def env_scalar(self, t: float) -> Tuple[float, float]:
        if t < 0.0 or t > self.duration:
            return (0.0, 0.0)
        if self.shape == 'square':
            return (1.0, 0.0)
        if self.shape == 'gaussian':
            sigma = self.duration / self.n_sigma
            tc = self.duration / 2
            g = math.exp(-(t - tc) ** 2 / (2 * sigma ** 2))
            g0 = math.exp(-tc ** 2 / (2 * sigma ** 2))
            return ((g - g0) / (1 - g0), -(t - tc) / sigma ** 2 * g / (1 - g0))
        if self.shape == 'gaussian_square':
            r = self.rise
            sigma = r / 2
            if t < r:
                g = math.exp(-(t - r) ** 2 / (2 * sigma ** 2))
                de = -(t - r) / sigma ** 2 * g
                return ((g - math.exp(-2)) / (1 - math.exp(-2)), de / (1 - math.exp(-2))) if self.smooth_edges else (g, de)
            if t > self.duration - r:
                g = math.exp(-(t - (self.duration - r)) ** 2 / (2 * sigma ** 2))
                de = -(t - (self.duration - r)) / sigma ** 2 * g
                return ((g - math.exp(-2)) / (1 - math.exp(-2)), de / (1 - math.exp(-2))) if self.smooth_edges else (g, de)
            return (1.0, 0.0)
        raise ValueError(self.shape)

@dataclass
class Segment:
    pulses: List[Pulse] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((p.start + p.duration for p in self.pulses), default=0.0)
SeqElem = Union[Segment, float]

class Simulator:

    def __init__(self, dev: DeviceConfig, solver_options: Optional[dict]=None, *, mode: str='fast'):
        if mode not in {'fast', 'exact', 'qutip'}:
            raise ValueError("mode must be 'fast', 'exact', or 'qutip'")
        self.dev = dev
        solver_options = dict(solver_options or {})
        self.noise_order = int(solver_options.pop('noise_order', 5))
        self.noise_max_order = int(solver_options.pop('noise_max_order', 65))
        self.noise_tolerance = float(solver_options.pop('noise_tolerance', 2e-06))
        if not 3 <= self.noise_order < self.noise_max_order <= 513 or not 0 < self.noise_tolerance < 0.001:
            raise ValueError('Invalid slow-noise quadrature settings')
        from .hardware import validate_hardware
        validate_hardware(dev)
        self.exact_flat = bool(solver_options.pop('exact_flat', False))
        self.exact_idle = bool(solver_options.pop('exact_idle', False))
        self._full_flat_cache = {}
        self.n = dev.n
        self.dims = [q.n_levels for q in dev.qubits]
        self.D = int(np.prod(self.dims))
        self.opts = {'nsteps': 500000, 'atol': 1e-11, 'rtol': 1e-09, 'store_states': False, 'store_final_state': True}
        if solver_options:
            self.opts.update(solver_options)
        self._build_operators()
        self._build_static()
        self._liou_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._dressed: Optional[Tuple[np.ndarray, List[Tuple[int, ...]]]] = None
        self.mode = mode
        self._prop_cache: Dict[tuple, np.ndarray] = {}
        self._prop_subdur: Dict[tuple, list] = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'uncacheable': 0}

    def _embed(self, op: qt.Qobj, k: int) -> qt.Qobj:
        ops = [qt.qeye(d) for d in self.dims]
        ops[k] = op
        return qt.tensor(ops)

    def _build_operators(self):
        if 'cosine_transmon' in self.dev.physics:
            from .cosine_transmon import operators
            operators(self)
            return
        self.a = [self._embed(qt.destroy(d), k) for k, d in enumerate(self.dims)]
        self.ad = [op.dag() for op in self.a]
        self.num = [self.ad[k] * self.a[k] for k in range(self.n)]
        self.X = [self.a[k] + self.ad[k] for k in range(self.n)]
        self.Y = [1j * (self.a[k] - self.ad[k]) for k in range(self.n)]

    def _build_static(self):
        if 'cosine_transmon' in self.dev.physics:
            from .cosine_transmon import static
            static(self)
            return
        dev = self.dev
        H_lin = 0
        H_anh = 0
        for k, qb in enumerate(dev.qubits):
            H_lin += TWO_PI * qb.freq * self.num[k]
            H_anh += TWO_PI * qb.anharm / 2 * self.num[k] * (self.num[k] - 1)
        H_c = 0
        self.coup_terms = []
        for c in dev.couplings:
            op = self.ad[c.i] * self.a[c.j]
            op_re = op + op.dag()
            op_im = 1j * (op - op.dag())
            H_c += TWO_PI * c.g * op_re
            self.coup_terms.append((c.i, c.j, TWO_PI * c.g, op_re, op_im))
        self.H_lin, self.H_anh, self.H_c = (H_lin, H_anh, H_c)
        self.H_static = H_lin + H_anh + H_c
        self.c_ops = []
        for k, qb in enumerate(dev.qubits):
            if qb.T1 and np.isfinite(qb.T1):
                self.c_ops.append(np.sqrt(1.0 / qb.T1) * self.a[k])
            if qb.Tphi and np.isfinite(qb.Tphi):
                self.c_ops.append(np.sqrt(2.0 / qb.Tphi) * self.num[k])
        self.w_frame = np.array([TWO_PI * qb.freq for qb in dev.qubits])

    def dressed(self) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
        if self._dressed is None:
            E, V = la.eigh(self.H_static.full())
            labels_all = list(np.ndindex(*self.dims))
            overlap = np.abs(V) ** 2
            assigned_vec = {}
            order = np.argsort(-overlap, axis=None)
            used_bare, used_vec = (set(), set())
            for flat in order:
                b, v = np.unravel_index(flat, overlap.shape)
                if b in used_bare or v in used_vec:
                    continue
                assigned_vec[b] = v
                used_bare.add(b)
                used_vec.add(v)
                if len(used_bare) == self.D:
                    break
            cols = [assigned_vec[b] for b in range(self.D)]
            Vd = V[:, cols]
            self._dressed = (Vd, labels_all)
            self.dressed_energies = E[cols]
        return self._dressed

    def dressed_frequency(self, k: int, spectators: Optional[Dict[int, int]]=None) -> float:
        self.dressed()
        spec = {j: 0 for j in range(self.n)}
        if spectators:
            spec.update(spectators)
        lab0 = list((spec[j] for j in range(self.n)))
        lab0[k] = 0
        lab1 = list(lab0)
        lab1[k] = 1
        i0 = np.ravel_multi_index(tuple(lab0), self.dims)
        i1 = np.ravel_multi_index(tuple(lab1), self.dims)
        return float((self.dressed_energies[i1] - self.dressed_energies[i0]) / TWO_PI)

    def zz(self, i: int, j: int) -> float:
        return self.dressed_frequency(i, {j: 1}) - self.dressed_frequency(i, {j: 0})

    def driven_energies(self, drives: List[Tuple[int, float, float, float]]):
        from .hardware import require_legacy_segments
        require_legacy_segments(self.dev)
        w_ref = TWO_PI * drives[0][1]
        H = self._static_frame_H(w_ref)
        C = self.dev.crosstalk_matrix()
        for line, f, amp, ph in drives:
            Om0 = TWO_PI * self.dev.qubits[line].drive_scale * amp
            for j in range(self.n):
                cj = complex(C[j, line])
                if cj == 0:
                    continue
                c = 0.5 * Om0 * cj * cmath.exp(1j * ph)
                H = H + c.real * self.X[j] + c.imag * self.Y[j]
        E, V = la.eigh(H.full())
        Vd, _ = self.dressed()
        overlap = np.abs(Vd.conj().T @ V) ** 2
        order = np.argsort(-overlap, axis=None)
        assign = {}
        used_b, used_v = (set(), set())
        for flat in order:
            b, v = np.unravel_index(flat, overlap.shape)
            if b in used_b or v in used_v:
                continue
            assign[b] = v
            used_b.add(b)
            used_v.add(v)
            if len(used_b) == self.D:
                break
        Ed = np.array([E[assign[b]] for b in range(self.D)])
        return (Ed, w_ref)

    def driven_frequency(self, k: int, drives, spectators: Optional[Dict[int, int]]=None) -> float:
        Ed, w_ref = self.driven_energies(drives)
        spec = {j: 0 for j in range(self.n)}
        if spectators:
            spec.update(spectators)
        lab0 = list((spec[j] for j in range(self.n)))
        lab0[k] = 0
        lab1 = list(lab0)
        lab1[k] = 1
        i0 = np.ravel_multi_index(tuple(lab0), self.dims)
        i1 = np.ravel_multi_index(tuple(lab1), self.dims)
        return float((Ed[i1] - Ed[i0] + w_ref) / TWO_PI)

    def driven_zz(self, i: int, j: int, drives) -> float:
        return self.driven_frequency(i, drives, {j: 1}) - self.driven_frequency(i, drives, {j: 0})

    def ground(self) -> qt.Qobj:
        Vd, _ = self.dressed()
        psi = qt.Qobj(Vd[:, 0], dims=[self.dims, [1] * self.n])
        return psi * psi.dag()

    def populations(self, rho: qt.Qobj) -> np.ndarray:
        Vd, _ = self.dressed()
        R = rho.full()
        p = np.real(np.einsum('ij,jk,ki->i', Vd.conj().T, R, Vd))
        p = np.clip(p, 0, None)
        return p.reshape(self.dims)

    def _Ntot(self) -> np.ndarray:
        if not hasattr(self, '_ntot'):
            self._ntot = sum((np.real(np.diag(self.num[k].full())) for k in range(self.n)))
        return self._ntot

    def _R(self, t: float, w_ref: float) -> np.ndarray:
        return np.exp(1j * w_ref * self._Ntot() * t)

    def to_frame(self, rho_lab: qt.Qobj, t: float, w_ref: float) -> qt.Qobj:
        r = self._R(t, w_ref)
        M = rho_lab.full()
        return qt.Qobj(r[:, None] * M * r.conj()[None, :], dims=rho_lab.dims)

    def to_lab(self, rho_frame: qt.Qobj, t: float, w_ref: float) -> qt.Qobj:
        r = self._R(t, w_ref).conj()
        M = rho_frame.full()
        return qt.Qobj(r[:, None] * M * r.conj()[None, :], dims=rho_frame.dims)
    SECULAR_ABOVE = 32

    def _liouvillian_eig(self):
        if self._liou_cache is None:
            L = qt.liouvillian(self.H_static, self.c_ops).full()
            w, V = la.eig(L)
            Vinv = la.inv(V)
            self._liou_cache = (w, V, Vinv)
        return self._liou_cache

    def _secular_blocks(self, cluster_width: float=TWO_PI * 0.03):
        if getattr(self, '_sec_cache', None) is not None:
            return self._sec_cache
        Vd, _ = self.dressed()
        E = self.dressed_energies
        D = self.D
        c_d = [Vd.conj().T @ c.full() @ Vd for c in self.c_ops]
        cdc = [c.conj().T @ c for c in c_d]
        idx = np.arange(D * D)
        m_all = idx % D
        n_all = idx // D
        bohr = E[m_all] - E[n_all]
        order = np.argsort(bohr)
        groups, cur = ([], [order[0]])
        for a, b in zip(order[:-1], order[1:]):
            if abs(bohr[b] - bohr[a]) < cluster_width:
                cur.append(b)
            else:
                groups.append(np.array(cur))
                cur = [b]
        groups.append(np.array(cur))
        blocks = []
        for g in groups:
            m = m_all[g]
            n = n_all[g]
            dm = m[:, None] == m[None, :]
            dn = n[:, None] == n[None, :]
            sub = np.diag(-1j * (E[m] - E[n])).astype(complex)
            for c, cc in zip(c_d, cdc):
                sub += c[np.ix_(m, m)] * c[np.ix_(n, n)].conj()
                sub -= 0.5 * cc[np.ix_(m, m)] * dn
                sub -= 0.5 * dm * cc[np.ix_(n, n)].conj()
            w, V = la.eig(sub)
            blocks.append((g, w, V, la.inv(V)))
        self._sec_cache = (Vd, blocks)
        return self._sec_cache

    def _qutip_segment_options(self, duration: float) -> dict:
        opts = self.opts.copy()
        opts['nsteps'] = max(opts['nsteps'], math.ceil(duration * 1000))
        return opts

    def idle(self, rho_lab: qt.Qobj, tau: float) -> qt.Qobj:
        from .hardware import require_legacy_segments
        require_legacy_segments(self.dev)
        if tau <= 0:
            return rho_lab
        if self.mode == 'qutip' and self.exact_idle:
            from .sim_engine import Engine
            if not hasattr(self, '_exact_idle_engine'):
                self._exact_idle_engine = Engine(self.dev, self.opts)
            return self._exact_idle_engine.idle(rho_lab, tau)
        if self.mode == 'qutip':
            w_ref = self.w_frame[0]
            res = qt.mesolve(self._static_frame_H(w_ref), rho_lab, [0.0, tau], c_ops=self.c_ops, options=self._qutip_segment_options(tau))
            return self.to_lab(res.final_state, tau, w_ref)
        if self.D <= self.SECULAR_ABOVE:
            w, V, Vinv = self._liouvillian_eig()
            vec = qt.operator_to_vector(rho_lab).full().ravel()
            out = V @ (np.exp(w * tau) * (Vinv @ vec))
            M = out.reshape((self.D, self.D), order='F')
        else:
            Vd, blocks = self._secular_blocks()
            R = Vd.conj().T @ rho_lab.full() @ Vd
            vec = R.reshape(-1, order='F').astype(complex)
            out = np.empty_like(vec)
            for g, w, V, Vinv in blocks:
                out[g] = V @ (np.exp(w * tau) * (Vinv @ vec[g]))
            M = out.reshape((self.D, self.D), order='F')
            M = Vd @ M @ Vd.conj().T
        M = 0.5 * (M + M.conj().T)
        return qt.Qobj(M, dims=rho_lab.dims)

    def idle_secular(self, rho_lab: qt.Qobj, tau: float) -> qt.Qobj:
        saved = self.SECULAR_ABOVE
        self.SECULAR_ABOVE = 0
        try:
            return self.idle(rho_lab, tau)
        finally:
            self.SECULAR_ABOVE = saved

    def _static_frame_H(self, w_ref: float) -> qt.Qobj:
        H = self.H_anh + self.H_c
        for k in range(self.n):
            H = H + (self.w_frame[k] - w_ref) * self.num[k]
        return H

    def _drive_terms(self, seg: Segment, t0: float, w_ref: float):
        from .parametric import CouplerSegment, drive_terms
        if isinstance(seg, CouplerSegment):
            return drive_terms(self, seg, t0)
        dev = self.dev
        C = dev.crosstalk_matrix()
        terms = []
        for p in seg.pulses:
            qb_line = dev.qubits[p.line]
            Om0 = TWO_PI * qb_line.drive_scale * p.amp
            wd = TWO_PI * p.freq
            dw = wd - w_ref
            for j in range(self.n):
                cj = complex(C[j, p.line]) if p.source_weights is None else sum((C[j, line] * self.dev.qubits[line].drive_scale / self.dev.qubits[p.line].drive_scale * complex(*weight) for line, weight in enumerate(p.source_weights)))
                if cj == 0:
                    continue
                alpha = TWO_PI * p.drag_anharm
                if p.drag != 0 and (not np.isfinite(alpha) or alpha == 0):
                    raise ValueError('Nonzero DRAG requires a finite, nonzero source drag_anharm')
                tstart = t0 + p.start

                def make(p=p, Om0=Om0, dw=dw, cj=cj, alpha=alpha, tstart=tstart):

                    def coef(t):
                        e, de = p.env_scalar(t - tstart)
                        quadrature = p.drag * de / alpha if p.drag != 0 else 0.0
                        om = Om0 * cj * complex(e, quadrature)
                        return 0.5 * om * cmath.exp(1j * (dw * t + p.phase))
                    if dw == 0.0 and p.drag == 0.0 and (cj.imag == 0.0) and (p.phase == 0.0):
                        return (lambda t: coef(t).real, None)
                    return (lambda t: coef(t).real, lambda t: coef(t).imag)
                cx, cy = make()
                terms.append([self.X[j], cx])
                if cy is not None:
                    terms.append([self.Y[j], cy])
        return terms

    @staticmethod
    def _is_flat_top(seg: Segment) -> bool:
        from .parametric import CouplerSegment
        if isinstance(seg, CouplerSegment):
            return False
        p0 = seg.pulses[0]
        return all((p.shape == 'gaussian_square' and p.freq == p0.freq and (p.start == 0.0) and (p.duration == p0.duration) and (p.rise == p0.rise) and (p.duration > 2 * p.rise + 1.0) for p in seg.pulses))

    @staticmethod
    def _is_constant(seg: Segment) -> bool:
        from .parametric import CouplerSegment
        if isinstance(seg, CouplerSegment):
            return False
        dur = seg.duration
        return all((p.shape == 'square' and p.freq == seg.pulses[0].freq and (p.start == 0.0) and (p.duration == dur) for p in seg.pulses))

    def _constant_H(self, seg: Segment, w_ref: float) -> qt.Qobj:
        H = self._static_frame_H(w_ref)
        C = self.dev.crosstalk_matrix()
        for p in seg.pulses:
            Om0 = TWO_PI * self.dev.qubits[p.line].drive_scale * p.amp
            for j in range(self.n):
                cj = complex(C[j, p.line]) if p.source_weights is None else sum((C[j, line] * self.dev.qubits[line].drive_scale / self.dev.qubits[p.line].drive_scale * complex(*weight) for line, weight in enumerate(p.source_weights)))
                if cj == 0:
                    continue
                c = 0.5 * Om0 * cj * cmath.exp(1j * p.phase)
                H = H + c.real * self.X[j] + c.imag * self.Y[j]
        return H

    def pulse(self, rho_lab: qt.Qobj, seg: Segment, t0: float) -> qt.Qobj:
        from .hardware import require_legacy_segments
        require_legacy_segments(self.dev)
        dur = seg.duration
        if dur <= 0:
            return rho_lab
        w_ref = TWO_PI * seg.pulses[0].freq
        rho_f = self.to_frame(rho_lab, t0, w_ref)
        if self.mode == 'qutip' and self.exact_flat and (self.D <= 25) and self._is_flat_top(seg):
            rise = seg.pulses[0].rise
            H = [self._static_frame_H(w_ref)] + self._drive_terms(seg, t0, w_ref)
            opts = self._qutip_segment_options(rise)
            opts['max_step'] = min(opts.get('max_step', 0) or rise / 4, rise / 4)
            rho_edge = qt.mesolve(H, rho_f, [t0, t0 + rise], c_ops=self.c_ops, options=opts).final_state
            key = (self._segment_key(seg), seg.pulses[0].phase)
            propagator = self._full_flat_cache.get(key)
            if propagator is None:
                L = qt.liouvillian(self._constant_H(seg, w_ref), self.c_ops).full()
                propagator = expm(L * (dur - 2 * rise))
                if len(self._full_flat_cache) >= 4:
                    self._full_flat_cache.clear()
                self._full_flat_cache[key] = propagator
            middle = apply_propagator(propagator, rho_edge.full().reshape(-1, order='F'))
            rho_middle = qt.Qobj(middle.reshape((self.D, self.D), order='F'), dims=rho_f.dims)
            rho_end = qt.mesolve(H, rho_middle, [t0 + dur - rise, t0 + dur], c_ops=self.c_ops, options=opts).final_state
            return self.to_lab(rho_end, t0 + dur, w_ref)
        if (self.mode != 'qutip' or (self.exact_flat and self.D <= 25)) and self._is_constant(seg):
            H = self._constant_H(seg, w_ref)
            L = qt.liouvillian(H, self.c_ops).full()
            vec = qt.operator_to_vector(rho_f).full().ravel()
            out = apply_propagator(expm(L * dur), vec)
            M = out.reshape((self.D, self.D), order='F')
            rho_end = qt.Qobj(0.5 * (M + M.conj().T), dims=rho_f.dims)
        else:
            H = [self._static_frame_H(w_ref)] + self._drive_terms(seg, t0, w_ref)
            opts = self.opts.copy()
            if self.mode == 'qutip':
                opts = self._qutip_segment_options(dur)
                features = [p.duration / p.n_sigma if p.shape == 'gaussian' else p.rise if p.shape == 'gaussian_square' else p.duration for p in seg.pulses if p.duration > 0]
                step = min(features) / 4
                opts['max_step'] = min(opts.get('max_step', 0) or step, step)
            res = qt.mesolve(H, rho_f, [t0, t0 + dur], c_ops=self.c_ops, options=opts)
            rho_end = res.final_state
        return self.to_lab(rho_end, t0 + dur, w_ref)

    @staticmethod
    def _segment_key(seg: Segment) -> tuple:
        ph0 = seg.pulses[0].phase
        return tuple(((p.line, p.freq, p.amp, p.duration, round(p.phase - ph0, 12), p.shape, p.drag, p.start, p.rise, p.n_sigma, p.drag_anharm, p.smooth_edges, None if p.source_weights is None else tuple((tuple(w) for w in p.source_weights))) for p in seg.pulses))
    SUBSTEP_NS = 10.0
    SUBSTEP_FLAT_MAX = 40.0

    def _flat_substep(self, seg: Segment) -> float:
        om = max((TWO_PI * self.dev.qubits[p.line].drive_scale * abs(p.amp) for p in seg.pulses)) + 1e-12
        return float(np.clip(self.SUBSTEP_NS * 0.125 / om, self.SUBSTEP_NS, self.SUBSTEP_FLAT_MAX))

    def _unitaries(self, seg: Segment, w_ref: float) -> List[np.ndarray]:
        key = self._segment_key(seg)
        Us = self._prop_cache.get(key)
        if Us is not None:
            self.cache_stats['hits'] += 1
            return Us
        self.cache_stats['misses'] += 1
        dur = seg.duration
        nsub = max(1, int(np.ceil(dur / self.SUBSTEP_NS)))
        edges = np.linspace(0.0, dur, nsub + 1)
        ph0 = seg.pulses[0].phase
        if ph0 != 0.0:
            seg = Segment([Pulse(**{**p.__dict__, 'phase': p.phase - ph0}) for p in seg.pulses])
        if self._is_constant(seg):
            nsub = max(1, int(np.ceil(dur / self._flat_substep(seg))))
            Hc = self._constant_H(seg, w_ref).full()
            U1 = expm(-1j * Hc * (dur / nsub))
            Us = [U1] * nsub
            self._prop_cache[key] = Us
            return Us
        H = [self._static_frame_H(w_ref)] + self._drive_terms(seg, 0.0, w_ref)
        if self._is_flat_top(seg):
            rise = seg.pulses[0].rise
            n_edge = max(1, int(np.ceil(rise / self.SUBSTEP_NS)))
            flat = dur - 2 * rise
            n_flat = max(1, int(np.ceil(flat / self._flat_substep(seg))))
            opts = {'nsteps': self.opts['nsteps'], 'atol': 1e-10, 'rtol': 1e-08, 'store_states': True, 'method': 'dop853'}
            I = qt.Qobj(np.eye(self.D, dtype=complex), dims=self.H_static.dims)
            e1 = np.linspace(0.0, rise, n_edge + 1)
            r1 = qt.sesolve(H, I, e1, options=opts)
            U1 = [s.full() for s in r1.states]
            Us = [U1[k + 1] @ U1[k].conj().T for k in range(n_edge)]
            flat_seg = Segment([Pulse(**{**p.__dict__, 'shape': 'square', 'duration': flat, 'start': 0.0}) for p in seg.pulses])
            Hc = self._constant_H(flat_seg, w_ref).full()
            Us += [expm(-1j * Hc * (flat / n_flat))] * n_flat
            e2 = np.linspace(dur - rise, dur, n_edge + 1)
            r2 = qt.sesolve(H, I, e2, options=opts)
            U2 = [s.full() for s in r2.states]
            Us += [U2[k + 1] @ U2[k].conj().T for k in range(n_edge)]
            self._prop_cache[key] = Us
            self._prop_subdur[key] = [rise / n_edge] * n_edge + [flat / n_flat] * n_flat + [rise / n_edge] * n_edge
            return Us
        opts = {'nsteps': self.opts['nsteps'], 'atol': 1e-10, 'rtol': 1e-08, 'store_states': True, 'method': 'dop853'}
        I = qt.Qobj(np.eye(self.D, dtype=complex), dims=self.H_static.dims)
        res = qt.sesolve(H, I, edges, options=opts)
        Ufull = [s.full() for s in res.states]
        Us = []
        for k in range(nsub):
            Us.append(Ufull[k + 1] @ Ufull[k].conj().T)
        self._prop_cache[key] = Us
        return Us

    def _secular_ops(self):
        if getattr(self, '_sec_ops', None) is None:
            import scipy.sparse as sp
            Vd, blocks = self._secular_blocks()
            D2 = self.D * self.D
            rows, cols, vals, rows_i, cols_i, vals_i = ([], [], [], [], [], [])
            w_all = np.zeros(D2, dtype=complex)
            for g, w, V, Vinv in blocks:
                gi, gj = np.meshgrid(g, g, indexing='ij')
                rows.append(gi.ravel())
                cols.append(gj.ravel())
                vals.append(V.ravel())
                rows_i.append(gi.ravel())
                cols_i.append(gj.ravel())
                vals_i.append(Vinv.ravel())
                w_all[g] = w
            Vs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D2, D2))
            Vis = sp.csr_matrix((np.concatenate(vals_i), (np.concatenate(rows_i), np.concatenate(cols_i))), shape=(D2, D2))
            E = self.dressed_energies
            self._sec_ops = (Vd, Vs, Vis, w_all, E)
        return self._sec_ops

    def _local_diss(self, k: int, tau: float) -> np.ndarray:
        key = (k, tau)
        S = getattr(self, '_ld_cache', {}).get(key)
        if S is not None:
            return S
        d = self.dims[k]
        qb = self.dev.qubits[k]
        a = qt.destroy(d)
        n = a.dag() * a
        cs = []
        if qb.T1 and np.isfinite(qb.T1):
            cs.append(np.sqrt(1.0 / qb.T1) * a)
        if qb.Tphi and np.isfinite(qb.Tphi):
            cs.append(np.sqrt(2.0 / qb.Tphi) * n)
        L = qt.liouvillian(0 * n, cs).full() if cs else np.zeros((d * d, d * d), dtype=complex)
        S = expm(L * tau)
        if not hasattr(self, '_ld_cache'):
            self._ld_cache = {}
        self._ld_cache[key] = S
        return S

    def _dissipate(self, M: np.ndarray, tau: float) -> np.ndarray:
        n = self.n
        dims = self.dims
        T = M.reshape(dims + dims)
        for k in range(n):
            S = self._local_diss(k, tau)
            d = dims[k]
            S4 = S.reshape(d, d, d, d, order='F')
            T = np.moveaxis(T, (k, n + k), (0, 1))
            shp = T.shape
            T = (S4.reshape(d * d, d * d) @ T.reshape(d * d, -1)).reshape(shp)
            T = np.moveaxis(T, (0, 1), (k, n + k))
        return T.reshape(self.D, self.D)

    def pulse_fast(self, rho_lab: qt.Qobj, seg: Segment, t0: float) -> qt.Qobj:
        dur = seg.duration
        w_ref = TWO_PI * seg.pulses[0].freq
        Us = self._unitaries(seg, w_ref)
        subdur = self._prop_subdur.get(self._segment_key(seg), [dur / len(Us)] * len(Us))
        ph0 = seg.pulses[0].phase
        M = self._R(t0, w_ref)
        if ph0 != 0.0:
            M = M * np.exp(1j * ph0 * self._Ntot())
        rho = M[:, None] * rho_lab.full() * M.conj()[None, :]
        for U, dt in zip(Us, subdur):
            rho = self._dissipate(rho, dt / 2)
            rho = U @ rho @ U.conj().T
            rho = self._dissipate(rho, dt / 2)
        M = self._R(t0 + dur, w_ref).conj()
        if ph0 != 0.0:
            M = M * np.exp(-1j * ph0 * self._Ntot())
        rho = M[:, None] * rho * M.conj()[None, :]
        return qt.Qobj(0.5 * (rho + rho.conj().T), dims=rho_lab.dims)

    def _cacheable(self, seg: Segment) -> bool:
        from .parametric import CouplerSegment
        if isinstance(seg, CouplerSegment):
            return False
        f0 = seg.pulses[0].freq
        return all((p.freq == f0 for p in seg.pulses))

    def run(self, seq: List[SeqElem], rho0: Optional[qt.Qobj]=None, t0: float=0.0, return_state: bool=False):
        from .hardware import dynamics_enabled
        if dynamics_enabled(self.dev):
            from .extended_dynamics import run
            return run(self, seq, rho0, t0, return_state)
        from .sim_client import enabled, solve_many
        if enabled(self):
            return next(solve_many(self, [seq], rho0=rho0, t0=t0, return_state=return_state))
        rho = self.ground() if rho0 is None else rho0
        t = t0
        for el in seq:
            if isinstance(el, Segment):
                if self.mode == 'fast' and el.duration > 0 and self._cacheable(el):
                    rho = self.pulse_fast(rho, el, t)
                else:
                    if self.mode == 'fast' and el.duration > 0:
                        self.cache_stats['uncacheable'] += 1
                    rho = self.pulse(rho, el, t)
                t += el.duration
            else:
                rho = self.idle(rho, float(el))
                t += float(el)
        pops = self.populations(rho)
        if return_state:
            return (pops, rho, t)
        return pops

    def sequence_duration(self, seq):
        from .hardware import dynamics_enabled
        if dynamics_enabled(self.dev):
            from .extended_dynamics import duration
            return duration(self, seq)
        return sum((el.duration if isinstance(el, Segment) else float(el) for el in seq))

    def run_many(self, sequences, rho0=None, t0=0.0, return_state=False, rho0s=None):
        from .sim_client import enabled, solve_many
        if enabled(self):
            yield from solve_many(self, sequences, rho0=rho0, t0=t0, return_state=return_state, rho0s=rho0s)
        else:
            sequences = list(sequences)
            from .hardware import dynamics_enabled
            if rho0s is None and dynamics_enabled(self.dev):
                from .extended_dynamics import run_many
                yield from run_many(self, sequences, rho0=rho0, t0=t0, return_state=return_state)
                return
            initials = [rho0] * len(sequences) if rho0s is None else list(rho0s)
            if len(initials) != len(sequences):
                raise ValueError('rho0s must match sequences')
            for seq, initial in zip(sequences, initials):
                yield self.run(seq, rho0=initial, t0=t0, return_state=return_state)

    def marginal(self, pops: np.ndarray, k: int) -> np.ndarray:
        axes = tuple((j for j in range(self.n) if j != k))
        return pops.sum(axis=axes)
