from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .device import DeviceConfig, CalibrationStore, ideal_calibration
from .core import Simulator, Pulse, Segment, SeqElem
from .readout import ReadoutChain

@dataclass
class Ledger:
    shots: int = 0
    experiments: int = 0
    sim_time_ns: float = 0.0
    log: List[dict] = field(default_factory=list)

    def add(self, name: str, shots: int, duration_ns: float, **meta):
        self.shots += shots
        self.experiments += 1
        self.sim_time_ns += duration_ns * shots
        self.log.append({'experiment': name, 'shots': shots, 'duration_ns': duration_ns, **meta})

class Env:

    def __init__(self, dev: DeviceConfig, cal: Optional[CalibrationStore]=None, seed: int=0, solver_options: Optional[dict]=None, *, solver_mode: str='fast'):
        self.dev = dev
        if 'flux_coupler' in dev.physics:
            from .coupler_runtime import CouplerSimulator
            self.sim = CouplerSimulator(dev, solver_options, mode=solver_mode)
        else:
            self.sim = Simulator(dev, solver_options, mode=solver_mode)
        self.ro = ReadoutChain(dev, np.random.default_rng(seed))
        self.cal = cal if cal is not None else CalibrationStore()
        if 'flux_coupler' in dev.physics:
            from .flux_coupler import FluxCouplerCalibration
            from copy import deepcopy
            if not isinstance(self.cal, FluxCouplerCalibration):
                self.cal = FluxCouplerCalibration(**deepcopy(vars(self.cal)))
        self.ledger = Ledger()
        self.readout_overhead_ns = 1000.0
        self._hardware_key = None
        self.sync_physics()

    def sync_physics(self):
        from .nonlinear_readout import validate_amplitudes
        validate_amplitudes(self.dev, self.cal.readout_amplitude)
        self.ro.amplitudes = dict(self.cal.readout_amplitude)
        from .instruments import validate as validate_instruments
        validate_instruments(self.dev, self.cal)
        self.ro.electronics_cal = self.cal
        self.ro.electronics_durations = self.cal.readout_duration
        if 'flux_coupler' in self.dev.physics:
            from .coupler_runtime import CouplerSimulator
            unsupported = ('lo_freq', 'mixer_gain', 'mixer_skew', 'mixer_dc_i', 'mixer_dc_q', 'channel_delay', 'dd_sequence', 'dd_pulses', 'dd_offset', 'dd_min_duration', 'readout_weights')
            if any((getattr(self.cal, k) for k in unsupported)):
                raise ValueError('This coupler backend does not combine mixer, XY delay, installed DD, or temporal readout controls')
            key = self.dev.to_json()
            if key != self._hardware_key:
                self.sim = CouplerSimulator(self.dev, self.sim.opts, mode=self.sim.mode)
                self._hardware_key = key
            self.sim.cal = self.cal
            self.sim.bias()
            self.ro.dev = self.dev
            if self.dev.physics.get('readout', {}).get('nonlinear'):
                from copy import deepcopy
                from .nonlinear_readout import validate_readout
                self.ro.dev = deepcopy(self.dev)
                frequencies = self.sim.models[False].frequencies(self.sim.bias())
                for k, q in enumerate(self.ro.dev.qubits):
                    q.freq = float(frequencies[k])
                validate_readout(self.ro.dev)
            return
        if not self.dev.physics and self._hardware_key is None:
            from .hardware import apparatus_controls_active
            if not apparatus_controls_active(self.cal):
                return
        from .hardware import fingerprint, effective_device, runtime_controls
        key = fingerprint(self.dev, self.cal)
        if key == self._hardware_key:
            return
        dev = effective_device(self.dev, self.cal)
        old = self.sim
        self.sim = Simulator(dev, {**old.opts, 'exact_flat': old.exact_flat, 'exact_idle': old.exact_idle, 'noise_order': old.noise_order, 'noise_max_order': old.noise_max_order, 'noise_tolerance': old.noise_tolerance}, mode=old.mode)
        self.sim.hardware_controls = runtime_controls(self.cal)
        self.ro.dev = dev
        self._hardware_key = key

    def flux_segment(self, pulses, duration=None):
        if 'flux_coupler' not in self.dev.physics:
            raise ValueError('This device has no tunable coupler')
        from .flux_coupler import FluxSegment
        return FluxSegment(tuple(pulses), max((p.duration for p in pulses)) if duration is None else duration)

    def set_flux_controls(self, *, qubit_bias=None, coupler_bias=None, mix=None, inverse_tau=None, inverse_tail=None):
        if 'flux_coupler' not in self.dev.physics:
            raise ValueError('No tunable coupler')
        from .flux_coupler import flux_trace
        candidate = self.cal.copy()
        if qubit_bias is not None:
            candidate.flux_bias = dict(qubit_bias)
        if coupler_bias is not None:
            candidate.coupler_flux_bias = {0: coupler_bias}
        if mix is not None:
            candidate.flux_mix = {0: mix}
        if inverse_tau is not None:
            candidate.flux_inverse_tau = dict(inverse_tau)
        if inverse_tail is not None:
            candidate.flux_inverse_tail = dict(inverse_tail)
        flux_trace(self.dev, candidate, [], 1.0, 1.0)
        for name in ('flux_bias', 'coupler_flux_bias', 'flux_mix', 'flux_inverse_tau', 'flux_inverse_tail'):
            setattr(self.cal, name, getattr(candidate, name))
        self.sync_physics()

    def drive_pulse(self, k: int, freq: float, amp: float, duration: float, **kw) -> Pulse:
        self.sync_physics()
        kw.setdefault('source_weights', self.cal.source_mix.get(k))
        kw.setdefault('drag_anharm', self.cal.anharm.get(k, -0.33))
        return Pulse(line=self.cal.drive_line.get(k, k), freq=freq, amp=amp, duration=duration, **kw)

    def x_pulse(self, k: int, angle: float=np.pi, phase: float=0.0, freq: Optional[float]=None, amp: Optional[float]=None, duration: Optional[float]=None, drag: Optional[float]=None) -> Pulse:
        c = self.cal
        dur = c.pi_duration.get(k, 40.0) if duration is None else duration
        a = c.pi_amp.get(k, 0.5) * angle / np.pi if amp is None else amp
        return self.drive_pulse(k, freq=c.drive_freq[k] if freq is None else freq, amp=a, duration=dur, phase=phase, shape='gaussian', drag=c.drag.get(k, 0.0) if drag is None else drag)

    def y_pulse(self, k, angle=np.pi, **kw):
        return self.x_pulse(k, angle, phase=np.pi / 2, **kw)

    def x12_pulse(self, k: int, angle: float=np.pi, phase: float=0.0) -> Pulse:
        c = self.cal
        f12 = c.drive_freq[k] + c.anharm[k]
        return self.drive_pulse(k, freq=f12, amp=c.pi_amp[k] * angle / np.pi / np.sqrt(2), duration=c.pi_duration[k], phase=phase, shape='gaussian', drag=0.0)

    def cr_pulse(self, control: int, target: int, amp=None, duration=None, phase=None, rise=None, drag=None) -> Pulse:
        c, edge = (self.cal, (control, target))
        return self.drive_pulse(control, c.drive_freq[target], c.cr_amp[edge] if amp is None else amp, c.cr_duration[edge] if duration is None else duration, phase=c.cr_phase.get(edge, 0.0) if phase is None else phase, rise=c.cr_rise.get(edge, 20.0) if rise is None else rise, drag=c.cr_drag.get(edge, 0.0) if drag is None else drag, shape='gaussian_square')

    def cr_segment(self, control: int, target: int, *, rotary=False, **kw) -> Segment:
        edge = (control, target)
        p = self.cr_pulse(control, target, **kw)
        pulses = [p]
        tone = self.cal.cr_cancel.get(edge, 0j)
        if rotary:
            tone += self.cal.cr_rotary.get(edge, 0j)
        if tone != 0:
            pulses.append(self.drive_pulse(target, p.freq, abs(tone), p.duration, phase=p.phase + float(np.angle(tone)), rise=p.rise, shape=p.shape))
        return Segment(pulses)

    def ecr_sequence(self, control: int, target: int) -> List[SeqElem]:
        edge, c = ((control, target), self.cal)
        half = c.cr_duration[edge] / 2
        rise = c.cr_rise.get(edge, 20.0)
        if half <= 0 or rise <= 0 or 2 * rise > half:
            raise ValueError('ECR half duration must be positive and at least twice cr_rise')
        b = c.ecr_balance.get(edge, 0.0)
        if abs(b) >= 1:
            raise ValueError('ECR balance must lie strictly between -1 and 1')
        phase = c.cr_phase.get(edge, 0.0)
        first = self.cr_segment(control, target, amp=c.cr_amp[edge] * (1 + b), duration=half, phase=phase, rotary=True)
        second = self.cr_segment(control, target, amp=c.cr_amp[edge] * (1 - b), duration=half, phase=phase + np.pi + c.ecr_phase_skew.get(edge, 0.0), rotary=True)
        return [first, Segment([self.x_pulse(control)]), second, Segment([self.x_pulse(control)])]

    def sizzle_tones(self, i: int, j: int):
        edge, c = ((i, j), self.cal)
        a, b = c.sizzle_amp[edge]
        return [(i, c.sizzle_freq[edge], a, 0.0), (j, c.sizzle_freq[edge], b, c.sizzle_phase.get(edge, 0.0))]

    def sizzle_segment(self, i: int, j: int, duration=None) -> Segment:
        edge = (i, j)
        duration = self.cal.sizzle_duration[edge] if duration is None else duration
        rise = self.cal.sizzle_rise.get(edge, 60.0)
        if duration <= 0 or rise <= 0 or 2 * rise > duration:
            raise ValueError('siZZle duration must be positive and at least twice sizzle_rise')
        return Segment([self.drive_pulse(k, f, a, duration, phase=ph, rise=rise, shape='gaussian_square', smooth_edges=bool(self.cal.sizzle_smooth.get(edge, 0))) for k, f, a, ph in self.sizzle_tones(i, j)])

    def post_gate_pulse(self, edge: Tuple[int, int], k: int, angle=np.pi / 2, phase=0.0) -> Pulse:
        correction = self.cal.post_gate_phase.get(edge, (0.0, 0.0))[edge.index(k)]
        return self.x_pulse(k, angle, phase=phase + correction)

    def simulate(self, seq: List[SeqElem], rho0=None):
        self.sync_physics()
        return self.sim.run(seq, rho0=rho0, return_state=True)

    def measure(self, seq: List[SeqElem], shots: int, qubits: Optional[List[int]]=None, name: str='custom', rho0=None) -> Dict[int, np.ndarray]:
        self.sync_physics()
        pops, rho, t_end = self._run_acquisition(seq, rho0)
        return self._sample_simulation(pops, t_end, shots, qubits, name)

    def _run_acquisition(self, seq, rho0=None):
        cache = getattr(self, '_acquisition_cache', None)
        if cache is not None:
            return cache.run(self.sim, seq, rho0)
        return self.sim.run(seq, rho0=rho0, return_state=True)

    def measure_many(self, sequences, shots, qubits=None, name='custom', rho0=None):
        self.sync_physics()
        results = self.sim.run_many(sequences, rho0=rho0, return_state=True)
        try:
            for pops, rho, t_end in results:
                yield self._sample_simulation(pops, t_end, shots, qubits, name)
        finally:
            results.close()

    def _sample_simulation(self, pops, t_end, shots, qubits, name):
        durations = {k: self.cal.readout_duration.get(k, 1000.0) for k in range(self.dev.n)}
        measured = list(range(self.dev.n)) if qubits is None else qubits
        duration = t_end + max((durations[k] + self.cal.readout_time_of_flight.get(k, 0.0) for k in measured), default=0.0) + self.readout_overhead_ns
        if hasattr(self.ledger, 'check'):
            self.ledger.check(shots, duration)
        iq = self.ro.sample_shots(pops, self.cal.readout_freq, shots, qubits, durations, weights=self.cal.readout_weights)
        measured = list(range(self.dev.n)) if qubits is None else qubits
        self.ledger.add(name, shots, t_end + max((durations[k] + self.cal.readout_time_of_flight.get(k, 0.0) for k in measured), default=0.0) + self.readout_overhead_ns)
        self._last_pops = pops
        return iq

    def measure_traces(self, seq, shots, qubits=None, name='readout_trace', rho0=None):
        if 'flux_coupler' in self.dev.physics:
            raise ValueError('Temporal readout is not yet part of the flux-coupler observation model')
        self.sync_physics()
        pops, _, t_end = self._run_acquisition(seq, rho0)
        durations = {k: self.cal.readout_duration.get(k, 1000.0) for k in range(self.dev.n)}
        if hasattr(self.ledger, 'check'):
            measured = list(range(self.dev.n)) if qubits is None else qubits
            self.ledger.check(shots, t_end + max((durations[k] + self.cal.readout_time_of_flight.get(k, 0.0) for k in measured)) + self.readout_overhead_ns)
        traces = self.ro.sample_traces(pops, self.cal.readout_freq, shots, qubits, durations)
        measured = list(traces)
        self.ledger.add(name, shots, t_end + max((durations[k] + self.cal.readout_time_of_flight.get(k, 0.0) for k in measured)) + self.readout_overhead_ns)
        return traces

    def assign(self, iq: np.ndarray, k: int) -> np.ndarray:
        return self.ro.assign(iq, self.cal.iq_angle.get(k, 0.0), self.cal.iq_threshold.get(k, 0.0))

    def p1(self, iq: np.ndarray, k: int) -> float:
        return float(self.assign(iq, k).mean())

    def assign3(self, iq: np.ndarray, k: int) -> np.ndarray:
        means = self.cal.iq_means3.get(k)
        if means is None:
            raise ValueError(f'Missing iq_means3 for q{k}; run gmm3_readout or supply measured means')
        return self.ro.assign_nearest(iq, np.asarray(means))

    def p_levels(self, iq: np.ndarray, k: int) -> np.ndarray:
        lab = self.assign3(iq, k)
        return np.array([np.mean(lab == m) for m in range(3)])

    def budget(self) -> dict:
        return {'shots': self.ledger.shots, 'experiments': self.ledger.experiments, 'sim_time_us': self.ledger.sim_time_ns / 1000}

def env_with_ideal_calibration(dev: DeviceConfig, seed: int=0, **kw) -> Env:
    cal = ideal_calibration(dev)
    env = Env(dev, cal, seed=seed, **kw)
    for k in range(dev.n):
        ang, thr = env.ro.optimal_discriminator(k, cal.readout_freq[k])
        cal.iq_angle[k], cal.iq_threshold[k] = (ang, thr)
        cal.drive_freq[k] = env.sim.dressed_frequency(k)
        cal.iq_means3[k] = list(env.ro.iq_means(k, cal.readout_freq[k])[:3])
    return env
