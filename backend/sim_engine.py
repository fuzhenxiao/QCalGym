from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import qutip as qt
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
from .core import Simulator, Segment, Pulse
from .device import DeviceConfig, QubitParams, CouplingParams, ResonatorParams

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def state_to_wire(rho):
    if rho is None:
        return None
    a = rho.full()
    return {'real': a.real.tolist(), 'imag': a.imag.tolist(), 'dims': rho.dims}

def state_from_wire(value):
    if value is None:
        return None
    return qt.Qobj(np.asarray(value['real']) + 1j * np.asarray(value['imag']), dims=value['dims'])

def sequence_to_wire(seq):
    from .parametric import CouplerSegment
    return [{**asdict(el), 'segment_type': 'parametric'} if isinstance(el, CouplerSegment) else {'pulses': [asdict(p) for p in el.pulses]} if isinstance(el, Segment) else float(el) for el in seq]

def sequence_from_wire(seq):
    from .parametric import CouplerSegment
    result = []
    for el in seq:
        if not isinstance(el, dict):
            result.append(float(el))
        elif el.get('segment_type') == 'parametric':
            data = {k: v for k, v in el.items() if k != 'segment_type'}
            data['pulses'] = [Pulse(**p) for p in data['pulses']]
            data['edge'] = tuple(data['edge'])
            result.append(CouplerSegment(**data))
        else:
            if 'segment_type' in el:
                raise ValueError('Unknown segment type')
            result.append(Segment([Pulse(**p) for p in el['pulses']]))
    return result

def device_from_wire(value):
    d = dict(value)
    d['qubits'] = [QubitParams(**x) for x in d['qubits']]
    d['couplings'] = [CouplingParams(**x) for x in d['couplings']]
    d['resonators'] = [ResonatorParams(**x) for x in d['resonators']]
    if d.get('drive_crosstalk') is not None:
        d['drive_crosstalk'] = [[complex(*z) for z in row] for row in d['drive_crosstalk']]
    if 'physical_model' in d:
        from .parametric import CouplerDevice
        return CouplerDevice(**d)
    return DeviceConfig(**d)

class Engine(Simulator):

    def __init__(self, dev, options):
        super().__init__(dev, options, mode='qutip')
        self._backend_worker = True
        self.exact_idle = True
        self._idle_blocks = None
        self.prefixes = OrderedDict()
        self.prefix_bytes = 0
        self.stats = {'prefix_hits': 0, 'prefix_misses': 0, 'idle_calls': 0}

    def pulse(self, rho_lab, seg, t0):
        from .three_qubit_runtime import supported, pulse
        if supported(self.dev) and self.exact_flat and seg.pulses and self._is_flat_top(seg):
            return pulse(self, rho_lab, seg, t0)
        return super().pulse(rho_lab, seg, t0)

    def run(self, seq, rho0=None, t0=0.0, return_state=False):
        from .hardware import dynamics_enabled
        if dynamics_enabled(self.dev):
            return Simulator.run(self, seq, rho0=rho0, t0=t0, return_state=return_state)
        value = self.solve(dict(seq=sequence_to_wire(seq), rho0=state_to_wire(rho0), t0=t0))
        pops = np.asarray(value['pops'])
        if return_state:
            return (pops, state_from_wire(value['rho']), value['t'])
        return pops

    def _drive_terms(self, seg, t0, w_ref):
        from .parametric import CouplerSegment, drive_terms
        if isinstance(seg, CouplerSegment):
            return drive_terms(self, seg, t0)
        C = self.dev.crosstalk_matrix()
        grouped = {}
        for p in seg.pulses:
            alpha = 2 * np.pi * p.drag_anharm
            if p.drag != 0 and (not np.isfinite(alpha) or alpha == 0):
                raise ValueError('Nonzero DRAG requires a finite, nonzero source drag_anharm')
            beta = p.drag / alpha if p.drag != 0 else 0.0
            dw = 2 * np.pi * p.freq - w_ref
            key = (p.duration, p.start, p.shape, p.rise, p.n_sigma, p.smooth_edges, dw, beta)
            if key not in grouped:
                grouped[key] = [p, 0 * self.X[0], 0 * self.X[0], dw, beta]
            row = grouped[key]
            for j in range(self.n):
                gain = C[j, p.line] * self.dev.qubits[p.line].drive_scale if p.source_weights is None else sum((C[j, k] * self.dev.qubits[k].drive_scale * complex(*weight) for k, weight in enumerate(p.source_weights)))
                c = np.pi * p.amp * gain * np.exp(1j * p.phase)
                row[1] += c.real * self.X[j] + c.imag * self.Y[j]
                row[2] += -c.imag * self.X[j] + c.real * self.Y[j]
        terms = []
        for p, A, B, dw, beta in grouped.values():

            def make(p=p, dw=dw, beta=beta, start=t0 + p.start):

                def coefficient(t):
                    e, de = p.env_scalar(t - start)
                    return complex(e, beta * de) * np.exp(1j * dw * t)
                if dw == 0 and beta == 0:
                    return (lambda t: p.env_scalar(t - start)[0], None)
                return (lambda t: coefficient(t).real, lambda t: coefficient(t).imag)
            real, imag = make()
            terms.append([A, real])
            if imag is not None:
                terms.append([B, imag])
        return terms

    def _build_idle_blocks(self):
        L = qt.liouvillian(self._static_frame_H(self.w_frame[0]), self.c_ops).to('CSR').data.as_scipy().tocsr()
        numbers = np.rint(self._Ntot()).astype(int)
        sectors = (numbers[:, None] - numbers[None, :]).reshape(-1, order='F')
        coo = L.tocoo()
        cross = coo.data[sectors[coo.row] != sectors[coo.col]]
        groups = [np.arange(self.D ** 2)] if np.any(cross != 0) else [np.flatnonzero(sectors == n) for n in np.unique(sectors)]
        blocks = []
        for indices in groups:
            block = L[indices][:, indices].tocsc()
            eig = None
            if len(indices) <= 1200:
                a = block.toarray()
                w, v = la.eig(a)
                try:
                    vi = la.inv(v)
                    condition = la.norm(v, np.inf) * la.norm(vi, np.inf)
                    residual = la.norm(block.dot(v) - v * w, np.inf) / max(1.0, la.norm(a, np.inf))
                    if condition < 1000000.0 and residual < 1e-11 and np.isfinite(vi).all():
                        eig = (w, v, vi)
                except la.LinAlgError:
                    pass
            blocks.append((indices, block, eig))
        self._idle_blocks = blocks

    def idle(self, rho_lab, tau):
        if tau <= 0:
            return rho_lab
        if self._idle_blocks is None:
            self._build_idle_blocks()
        vec = rho_lab.full().reshape(-1, order='F')
        out = np.empty_like(vec)
        for indices, block, eig in self._idle_blocks:
            part = vec[indices]
            if not np.any(part):
                out[indices] = 0
            elif eig is None:
                out[indices] = expm_multiply(block * tau, part)
            else:
                w, v, vi = eig
                coeff = np.einsum('ij,j->i', vi, part)
                out[indices] = np.einsum('ij,j->i', v, np.exp(w * tau) * coeff)
        if not np.isfinite(out).all():
            raise FloatingPointError('Nonfinite full-Lindblad idle propagation')
        self.stats['idle_calls'] += 1
        rho = qt.Qobj(out.reshape((self.D, self.D), order='F'), dims=rho_lab.dims)
        return self.to_lab(rho, tau, self.w_frame[0])

    def solve(self, job):
        started = time.monotonic()
        from .hardware import dynamics_enabled
        if dynamics_enabled(self.dev):
            pops, state, t = Simulator.run(self, sequence_from_wire(job['seq']), rho0=state_from_wire(job.get('rho0')), t0=float(job.get('t0', 0)), return_state=True)
            return {'pops': pops.tolist(), 'rho': state_to_wire(state), 't': t, 'stats': {'solve_seconds': time.monotonic() - started, 'full_sequence_hardware': True}}
        rho = state_from_wire(job.get('rho0'))
        rho = self.ground() if rho is None else rho
        t = float(job.get('t0', 0))
        key = digest({'rho0': job.get('rho0'), 't0': t})
        old_stats = dict(self.stats)
        for raw in job['seq']:
            key = digest([key, raw])
            cached = self.prefixes.get(key)
            if cached is not None:
                self.prefixes.move_to_end(key)
                rho, t = (cached[0].copy(), cached[1])
                self.stats['prefix_hits'] += 1
                continue
            self.stats['prefix_misses'] += 1
            el = sequence_from_wire([raw])[0]
            if isinstance(el, Segment):
                rho = self.pulse(rho, el, t)
                t += el.duration
            else:
                rho = self.idle(rho, el)
                t += el
            size = rho.full().nbytes
            self.prefixes[key] = (rho.copy(), t, size)
            self.prefix_bytes += size
            while self.prefix_bytes > 16 * 1024 ** 2 or len(self.prefixes) > 512:
                _, item = self.prefixes.popitem(last=False)
                self.prefix_bytes -= item[2]
        return {'pops': self.populations(rho).tolist(), 'rho': state_to_wire(rho), 't': t, 'stats': {**{k: self.stats[k] - old_stats[k] for k in self.stats}, 'solve_seconds': time.monotonic() - started}}
_ENGINES = OrderedDict()

def solve_job(spec, job):
    key = digest(spec)
    engine = _ENGINES.get(key)
    if engine is None:
        engine = Engine(device_from_wire(spec['device']), spec['options'])
        _ENGINES[key] = engine
        while len(_ENGINES) > 2:
            _ENGINES.popitem(last=False)
    _ENGINES.move_to_end(key)
    return engine.solve(job)
