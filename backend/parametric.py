from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from .device import DeviceConfig, CalibrationStore
from .core import Segment, Pulse
MODEL = 'three_parametric_v1'

def public_apparatus():
    return dict(physical_model=MODEL, description='Three Duffing transmons with weak residual exchange and a parametrically modulated effective coupler on each listed edge.', qubit_search_GHz=[4.7, 5.4], readout_search_GHz=[[7.0 + 0.08 * k - 0.015, 7.0 + 0.08 * k + 0.015] for k in range(3)], nominal_anharm_GHz=-0.33, nominal_pi_duration_ns=40.0, nominal_drag=0.5, coupler_note='Native CZ uses a full |11>-|02> excursion. A starting modulation frequency is abs(f_target + alpha_target - f_control). Calibrate frequency and nonzero return amplitude from measured scans; populations alone do not certify CZ phase. A 0.025 GHz peak with a sin-squared envelope suggests sqrt(2)/0.025 ns duration. CNOT wraps this CZ in finite target rotations.', cold_start_note='The calibration store is empty. These are shared engineering search ranges, not measured calibration values. Readout spectroscopy can precede qubit spectroscopy; Ramsey refines frequency using undriven waits.')

def enabled(dev):
    return isinstance(dev, CouplerDevice)

@dataclass
class CouplerDevice(DeviceConfig):
    physical_model: str = MODEL

    def __post_init__(self):
        if self.n != 3 or self.topology not in ('chain3', 'ring3'):
            raise ValueError('Parametric couplers are limited to three-qubit chain/ring')
        if self.physical_model != MODEL:
            raise ValueError('Unknown parametric model version')

@dataclass
class CouplerCalibration(CalibrationStore):
    coupler_amp: dict = field(default_factory=dict)
    coupler_freq: dict = field(default_factory=dict)
    coupler_duration: dict = field(default_factory=dict)
    coupler_phase: dict = field(default_factory=dict)

@dataclass
class CouplerSegment(Segment):
    edge: tuple = (0, 1)
    exchange_amp: float = 0.0
    modulation_freq: float = 0.0
    modulation_phase: float = 0.0

    def __post_init__(self):
        if len(self.pulses) != 1 or self.pulses[0].amp != 0.0 or self.pulses[0].start != 0.0:
            raise ValueError('Coupler segment requires exactly one zero-amplitude frame carrier')
        if not np.isfinite([self.duration, self.exchange_amp, self.modulation_freq, self.modulation_phase]).all() or self.duration <= 0.0 or self.modulation_freq <= 0.0:
            raise ValueError('Coupler waveform parameters must be finite with positive duration and frequency')

def segment(env, edge, *, amp=None, frequency=None, duration=None):
    cal = env.cal
    duration = cal.coupler_duration[edge] if duration is None else duration
    return CouplerSegment([Pulse(edge[0], env.cal.drive_freq[edge[0]], 0.0, duration)], tuple(edge), cal.coupler_amp[edge] if amp is None else amp, cal.coupler_freq[edge] if frequency is None else frequency, cal.coupler_phase.get(edge, 0.0))

def drive_terms(sim, seg, t0):
    if not isinstance(sim.dev, CouplerDevice):
        raise ValueError('Coupler pulses require the versioned three-qubit device')
    if not any((set(seg.edge) == set(e) for e in sim.dev.edges)):
        raise ValueError('Coupler pulse must address an installed edge')
    i, j = seg.edge
    exchange = sim.a[i].dag() * sim.a[j] + sim.a[j].dag() * sim.a[i]

    def coefficient(t):
        local = t - t0
        if local < 0.0 or local > seg.duration:
            return 0.0
        return 2 * np.pi * seg.exchange_amp * np.sin(np.pi * local / seg.duration) ** 2 * np.cos(2 * np.pi * seg.modulation_freq * local + seg.modulation_phase)
    return [[exchange, coefficient]]

def convert(env, seed=0):
    from .env import Env
    dev = CouplerDevice(**deepcopy(vars(env.dev)))
    dev.name += '_' + MODEL
    for coupling in dev.couplings:
        coupling.g /= 35.0
    cal = CouplerCalibration(**deepcopy(vars(env.cal)))
    for name in vars(cal):
        if name.startswith(('cr_', 'ecr_', 'sizzle_', 'hold_', 'idle_', 'cnot_', 'post_gate_')):
            getattr(cal, name).clear()
    for i, j in dev.edges:
        edge = (i, j)
        cal.coupler_amp[edge] = 0.025
        cal.coupler_duration[edge] = np.sqrt(2) / 0.025
        cal.coupler_freq[edge] = abs(dev.qubits[j].freq + dev.qubits[j].anharm - dev.qubits[i].freq)
        cal.coupler_phase[edge] = 0.0
        cal.post_gate_network_phase[edge] = [0.0] * 3
    return Env(dev, cal, seed=seed, solver_mode=env.sim.mode, solver_options={**env.sim.opts, 'exact_flat': env.sim.exact_flat, 'exact_idle': env.sim.exact_idle})

def compiled(env, kind, edge, frames=None):
    if kind == 'zz_idle':
        return ([1000.0], np.zeros(3))
    if kind not in ('cz', 'cnot'):
        raise ValueError('Unknown parametric logical gate')
    carried = np.zeros(3) if frames is None else np.asarray(frames)
    advance = np.asarray(env.cal.post_gate_network_phase.get(edge, [0.0] * 3))
    block = [segment(env, edge)]
    if kind == 'cnot':
        target = edge[1]
        block = [Segment([env.x_pulse(target, np.pi / 2, phase=carried[target] + np.pi / 2)]), *block, Segment([env.x_pulse(target, np.pi / 2, phase=carried[target] + advance[target] - np.pi / 2)])]
    return (block, advance)

def obsolete_control(name):
    return name.startswith(('cr_', 'ecr_', 'sizzle_', 'hold_', 'idle_', 'cnot_')) or name == 'post_gate_phase'

def control_bounds():
    return {'coupler_amp': (0.0, 0.04), 'coupler_freq': (0.05, 1.0), 'coupler_duration': (10.0, 150.0), 'coupler_phase': (-20.0, 20.0), 'pi_duration': (10.0, 80.0)}

def configure_scoring(env, cache_path=None):
    if not enabled(env.dev) or env.sim.mode != 'qutip':
        return env
    from .sim_engine import Engine
    env.sim = Engine(env.dev, {**env.sim.opts, 'method': 'dop853', 'exact_flat': True})
    env.sim.certified_short_layers = True
    env.sim.certified_long_operations = True
    if cache_path is not None:
        env.sim.channel_cache_path = cache_path
    return env
