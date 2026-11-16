from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import numpy as np
from .device import DeviceConfig, ResonatorParams
SIGMA0 = 0.15
T_REF = 1000.0
A0 = 1.0
KAPPA_E_FRAC = 0.8

def s21(res: ResonatorParams, f: np.ndarray, level: int) -> np.ndarray:
    shift = res.chi_levels[level] if res.chi_levels else res.chi * level
    fr = res.freq + shift
    ke = KAPPA_E_FRAC * res.kappa
    return 1.0 - ke / (res.kappa + 2j * (f - fr))

class ReadoutChain:

    def __init__(self, dev: DeviceConfig, rng: Optional[np.random.Generator]=None):
        self.dev = dev
        self.amplitudes = {}
        self.electronics_cal = None
        self.electronics_durations = {}
        self.rng = rng or np.random.default_rng(dev.seed + 12345)

    def resonator_spec(self, k: int, freqs: np.ndarray, pops_k: np.ndarray, n_avg: int=200) -> np.ndarray:
        res = self.dev.resonators[k]
        sig = np.zeros(len(freqs), dtype=complex)
        from .nonlinear_readout import apparatus
        if apparatus(self.dev, k) is None:
            for m, p in enumerate(pops_k):
                if p > 0:
                    sig += p * self.amplitudes.get(k, 1.0) * s21(res, freqs, m)
        else:
            means = np.array([self.iq_means(k, f) for f in freqs])
            for m, p in enumerate(pops_k):
                if p > 0:
                    sig += p * means[:, m]
        noise = SIGMA0 / np.sqrt(n_avg) * (self.rng.normal(size=len(freqs)) + 1j * self.rng.normal(size=len(freqs)))
        from .instruments import receiver
        scale, offset, gain = receiver(self.dev, self.electronics_cal, k, self.electronics_durations.get(k, 1000.0))
        return np.abs(scale * A0 * sig + offset + gain * noise)

    def waveforms(self, k, frequency, labels, jumps, edges, drive_duration):
        from .nonlinear_readout import waveforms
        return waveforms(self.dev, k, frequency, self.amplitudes.get(k, 1.0), labels, jumps, edges, drive_duration)

    def iq_means(self, k: int, f_ro: float) -> np.ndarray:
        res = self.dev.resonators[k]
        from .nonlinear_readout import pointer
        amplitude = self.amplitudes.get(k, 1.0)
        nonlinear = pointer(self.dev, k, f_ro, amplitude)
        if nonlinear is not None:
            return nonlinear
        return amplitude * np.array([A0 * s21(res, np.array([f_ro]), m)[0] for m in range(self.dev.qubits[k].n_levels)])

    def sample_shots(self, pops: np.ndarray, readout_freq: Dict[int, float], shots: int, qubits: Optional[List[int]]=None, integration_time: Optional[Dict[int, float]]=None, *, weights=None) -> Dict[int, np.ndarray]:
        from .hardware import temporal_enabled
        if temporal_enabled(self.dev):
            from .temporal_readout import integrate
            return integrate(self.sample_traces(pops, readout_freq, shots, qubits, integration_time), weights or {})
        dims = pops.shape
        flat = pops.ravel()
        flat = np.clip(flat, 0, None)
        flat = flat / flat.sum()
        idx = self.rng.choice(len(flat), size=shots, p=flat)
        labels = np.array(np.unravel_index(idx, dims)).T
        out = {}
        qubits = list(range(self.dev.n)) if qubits is None else qubits
        durations = {k: self.dev.resonators[k].t_int if integration_time is None else integration_time[k] for k in qubits}
        if any((not np.isfinite(t) or t <= 0 for t in durations.values())):
            raise ValueError('Readout integration times must be finite and positive')
        for k in qubits:
            res = self.dev.resonators[k]
            qb = self.dev.qubits[k]
            f_ro = readout_freq[k]
            means = self.iq_means(k, f_ro)
            m = labels[:, k]
            iq = means[m].astype(complex)
            one = np.where(m == 1)[0]
            if len(one) and np.isfinite(qb.T1):
                s = self.rng.exponential(qb.T1, size=len(one))
                w = np.clip(s / durations[k], 0, 1)
                iq[one] = w * means[1] + (1 - w) * means[0]
            out[k] = iq
        R = self.dev.readout_crosstalk
        if R is not None:
            for k in qubits:
                leak = np.zeros(shots, dtype=complex)
                for j in range(self.dev.n):
                    if j == k or R[k][j] == 0 or self.dev.resonators[j].feedline != self.dev.resonators[k].feedline:
                        continue
                    mj = self.iq_means(j, readout_freq[j])
                    leak += R[k][j] * (mj[labels[:, j]] - mj[0])
                out[k] = out[k] + leak
        for k in qubits:
            res = self.dev.resonators[k]
            sigma = SIGMA0 / res.snr_scale * np.sqrt(T_REF / durations[k])
            from .instruments import receiver
            scale, offset, gain = receiver(self.dev, self.electronics_cal, k, durations[k])
            out[k] = scale * out[k] + offset + gain * sigma * (self.rng.normal(size=shots) + 1j * self.rng.normal(size=shots))
        return out

    def sample_traces(self, pops, readout_freq, shots, qubits=None, integration_time=None):
        from .hardware import temporal_enabled
        if not temporal_enabled(self.dev):
            raise ValueError('This apparatus does not provide time-resolved readout')
        from .temporal_readout import sample
        qubits = list(range(self.dev.n)) if qubits is None else list(qubits)
        durations = {k: self.dev.resonators[k].t_int if integration_time is None else integration_time[k] for k in range(self.dev.n)}
        if any((not np.isfinite(t) or t <= 0 for t in durations.values())):
            raise ValueError('Readout durations must be finite and positive')
        return sample(self, pops, readout_freq, shots, qubits, durations)

    @staticmethod
    def assign(iq: np.ndarray, angle: float, threshold: float) -> np.ndarray:
        rot = iq * np.exp(-1j * angle)
        return (rot.real > threshold).astype(int)

    @staticmethod
    def assign_nearest(iq: np.ndarray, means: np.ndarray) -> np.ndarray:
        d = np.abs(iq[:, None] - np.asarray(means)[None, :])
        return np.argmin(d, axis=1)

    def optimal_discriminator(self, k: int, f_ro: float) -> Tuple[float, float]:
        m = self.iq_means(k, f_ro)
        d = m[1] - m[0]
        angle = float(np.angle(d))
        mid = (m[0] + m[1]) / 2 * np.exp(-1j * angle)
        return (angle, float(mid.real))
