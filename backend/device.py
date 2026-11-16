from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

@dataclass
class QubitParams:
    freq: float
    anharm: float
    T1: float
    Tphi: float
    drive_scale: float = 0.05
    n_levels: int = 3

@dataclass
class CouplingParams:
    i: int
    j: int
    g: float

@dataclass
class ResonatorParams:
    freq: float
    kappa: float
    chi: float
    chi_levels: Optional[List[float]] = None
    feedline: int = 0
    snr_scale: float = 1.0
    t_int: float = 1000.0

@dataclass
class DeviceConfig:
    qubits: List[QubitParams]
    couplings: List[CouplingParams]
    resonators: List[ResonatorParams]
    drive_crosstalk: Optional[List[List[complex]]] = None
    readout_crosstalk: Optional[List[List[float]]] = None
    seed: int = 0
    name: str = 'device'
    topology: str = ''
    physics: dict = field(default_factory=dict)

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(c.i, c.j) for c in self.couplings]

    def neighbours(self, k: int) -> List[int]:
        return sorted([j for i, j in self.edges if i == k] + [i for i, j in self.edges if j == k])

    @property
    def n(self) -> int:
        return len(self.qubits)

    def crosstalk_matrix(self) -> np.ndarray:
        if self.drive_crosstalk is None:
            return np.eye(self.n, dtype=complex)
        return np.array(self.drive_crosstalk, dtype=complex)

    def to_json(self) -> str:
        d = asdict(self)
        if self.drive_crosstalk is not None:
            d['drive_crosstalk'] = [[[c.real, c.imag] for c in row] for row in self.crosstalk_matrix()]
        return json.dumps(d, indent=2)

@dataclass
class CalibrationStore:
    drive_freq: Dict[int, float] = field(default_factory=dict)
    readout_freq: Dict[int, float] = field(default_factory=dict)
    readout_amplitude: Dict[int, float] = field(default_factory=dict)
    pi_amp: Dict[int, float] = field(default_factory=dict)
    pi_duration: Dict[int, float] = field(default_factory=dict)
    drag: Dict[int, float] = field(default_factory=dict)
    iq_angle: Dict[int, float] = field(default_factory=dict)
    iq_threshold: Dict[int, float] = field(default_factory=dict)
    cr_amp: Dict[Tuple[int, int], float] = field(default_factory=dict)
    cr_duration: Dict[Tuple[int, int], float] = field(default_factory=dict)
    cr_phase: Dict[Tuple[int, int], float] = field(default_factory=dict)
    anharm: Dict[int, float] = field(default_factory=dict)
    iq_means3: Dict[int, List[complex]] = field(default_factory=dict)
    readout_duration: Dict[int, float] = field(default_factory=dict)
    drive_line: Dict[int, int] = field(default_factory=dict)
    cr_rise: Dict[Tuple[int, int], float] = field(default_factory=dict)
    cr_drag: Dict[Tuple[int, int], float] = field(default_factory=dict)
    cr_cancel: Dict[Tuple[int, int], complex] = field(default_factory=dict)
    cr_rotary: Dict[Tuple[int, int], complex] = field(default_factory=dict)
    ecr_balance: Dict[Tuple[int, int], float] = field(default_factory=dict)
    ecr_phase_skew: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sizzle_freq: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sizzle_amp: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    sizzle_phase: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sizzle_duration: Dict[Tuple[int, int], float] = field(default_factory=dict)
    sizzle_rise: Dict[Tuple[int, int], float] = field(default_factory=dict)
    post_gate_phase: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    cnot_pre_phase: Dict[int, float] = field(default_factory=dict)
    cnot_frame_phase: Dict[int, float] = field(default_factory=dict)
    source_mix: Dict[int, list] = field(default_factory=dict)
    hold_freq: Dict[Tuple[int, int], float] = field(default_factory=dict)
    hold_amp: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    hold_phase: Dict[Tuple[int, int], float] = field(default_factory=dict)
    hold_rise: Dict[Tuple[int, int], float] = field(default_factory=dict)
    hold_frame: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    hold_enabled: Dict[Tuple[int, int], int] = field(default_factory=dict)
    hold_smooth: Dict[Tuple[int, int], int] = field(default_factory=dict)
    idle_active: Dict[int, int] = field(default_factory=dict)
    ecr_walsh: Dict[Tuple[int, int], int] = field(default_factory=dict)
    idle_tone_freq: Dict[int, float] = field(default_factory=dict)
    idle_tone_amp: Dict[int, float] = field(default_factory=dict)
    idle_tone_phase: Dict[int, float] = field(default_factory=dict)
    idle_tone_rise: Dict[int, float] = field(default_factory=dict)
    idle_tone_smooth: Dict[int, int] = field(default_factory=dict)
    sizzle_smooth: Dict[Tuple[int, int], int] = field(default_factory=dict)
    idle_frame_phase: Dict[int, float] = field(default_factory=dict)
    sizzle_network_amp: Dict[Tuple[int, int], list] = field(default_factory=dict)
    sizzle_network_phase: Dict[Tuple[int, int], list] = field(default_factory=dict)
    post_gate_network_phase: Dict[Tuple[int, int], list] = field(default_factory=dict)
    lo_freq: Dict[int, float] = field(default_factory=dict)
    mixer_gain: Dict[int, float] = field(default_factory=dict)
    mixer_skew: Dict[int, float] = field(default_factory=dict)
    mixer_dc_i: Dict[int, float] = field(default_factory=dict)
    mixer_dc_q: Dict[int, float] = field(default_factory=dict)
    channel_delay: Dict[int, float] = field(default_factory=dict)
    ampm_compensation: Dict[int, float] = field(default_factory=dict)
    dressing_freq: Dict[int, float] = field(default_factory=dict)
    dressing_amp: Dict[int, float] = field(default_factory=dict)
    flux_bias: Dict[int, float] = field(default_factory=dict)
    dd_sequence: Dict[int, str] = field(default_factory=dict)
    dd_pulses: Dict[int, int] = field(default_factory=dict)
    dd_offset: Dict[int, float] = field(default_factory=dict)
    dd_min_duration: Dict[int, float] = field(default_factory=dict)
    readout_weights: Dict[int, list] = field(default_factory=dict)
    readout_time_of_flight: Dict[int, float] = field(default_factory=dict)
    adc_offset_i: Dict[int, float] = field(default_factory=dict)
    adc_offset_q: Dict[int, float] = field(default_factory=dict)
    adc_gain: Dict[int, float] = field(default_factory=dict)

    def copy(self) -> 'CalibrationStore':
        import copy
        return copy.deepcopy(self)

def chain_device(n: int=3, seed: int=0, n_levels: int=3, **overrides) -> DeviceConfig:
    rng = np.random.default_rng(seed)
    base = [4.9, 5.02, 4.86, 4.98]
    qubits = []
    for k in range(n):
        f = base[k] + rng.normal(0, 0.005)
        qubits.append(QubitParams(freq=f, anharm=-0.33 + rng.normal(0, 0.005), T1=rng.uniform(40000, 90000), Tphi=rng.uniform(60000, 150000), drive_scale=rng.uniform(0.04, 0.06), n_levels=n_levels))
    couplings = [CouplingParams(k, k + 1, g=0.0035 + rng.normal(0, 0.0003)) for k in range(n - 1)]
    resonators = []
    for k in range(n):
        resonators.append(ResonatorParams(freq=7.0 + 0.08 * k + rng.normal(0, 0.003), kappa=0.0008, chi=-0.0004, chi_levels=None, feedline=0, snr_scale=1.0, t_int=1000.0))
    dev = DeviceConfig(qubits=qubits, couplings=couplings, resonators=resonators, seed=seed, name=f'chain{n}')
    for k, v in overrides.items():
        setattr(dev, k, v)
    return dev

def ideal_calibration(dev: DeviceConfig) -> CalibrationStore:
    if 'flux_coupler' in dev.physics:
        from .flux_coupler import FluxCouplerCalibration
        cal = FluxCouplerCalibration()
    else:
        cal = CalibrationStore()
    for k, qb in enumerate(dev.qubits):
        cal.drive_freq[k] = qb.freq
        cal.readout_freq[k] = dev.resonators[k].freq
        cal.pi_duration[k] = 40.0
        cal.pi_amp[k] = pi_amp_from_scale(qb.drive_scale, 40.0)
        cal.drag[k] = 0.0
        cal.iq_angle[k] = 0.0
        cal.iq_threshold[k] = 0.0
        cal.anharm[k] = qb.anharm
        cal.readout_duration[k] = dev.resonators[k].t_int
        cal.drive_line[k] = k
    return cal

def gaussian_area(duration: float, n_sigma: float=4.0) -> float:
    from scipy.special import erf
    sigma = duration / n_sigma
    t = np.linspace(-duration / 2, duration / 2, 4001)
    env = np.exp(-t ** 2 / (2 * sigma ** 2))
    env = env - env[0]
    env /= env.max()
    return float(np.trapezoid(env, t))

def pi_amp_from_scale(drive_scale: float, duration: float) -> float:
    return 1.0 / (2 * drive_scale * gaussian_area(duration))
TOPOLOGIES = {'pair': {'n': 2, 'edges': [(0, 1)]}, 'chain3': {'n': 3, 'edges': [(0, 1), (1, 2)]}, 'ring3': {'n': 3, 'edges': [(0, 1), (1, 2), (2, 0)]}, 'chain4': {'n': 4, 'edges': [(0, 1), (1, 2), (2, 3)]}, 'star4': {'n': 4, 'edges': [(0, 1), (0, 2), (0, 3)]}, 'ring4': {'n': 4, 'edges': [(0, 1), (1, 2), (2, 3), (3, 0)]}}
COLLISION_RULES = [('NN: f_i = f_j', 'nn', lambda fi, fj, ai, aj: abs(fi - fj), 0.017), ('NN: f_i = f_j + a_j/2 (2-photon)', 'nn', lambda fi, fj, ai, aj: min(abs(fi - fj - aj / 2), abs(fj - fi - ai / 2)), 0.004), ('NN: f01_i = f12_j', 'nn', lambda fi, fj, ai, aj: min(abs(fi - (fj + aj)), abs(fj - (fi + ai))), 0.03), ('NN: |Delta| > |alpha| (CR too weak / outside straddling)', 'nn', lambda fi, fj, ai, aj: max(0.0, min(abs(ai), abs(aj)) - abs(fi - fj)), 0.05), ('NNN: f_i = f_k (spectator = target)', 'nnn', lambda fi, fj, ai, aj: abs(fi - fj), 0.017), ('NNN: f01_i = f12_k', 'nnn', lambda fi, fj, ai, aj: min(abs(fi - (fj + aj)), abs(fj - (fi + ai))), 0.025)]

def check_collisions(dev: 'DeviceConfig') -> List[dict]:
    f = [q.freq for q in dev.qubits]
    a = [q.anharm for q in dev.qubits]
    nn = set((tuple(sorted(e)) for e in dev.edges))
    nnn = set()
    for k in range(dev.n):
        nb = dev.neighbours(k)
        for i in nb:
            for j in nb:
                if i < j and tuple(sorted((i, j))) not in nn:
                    nnn.add((i, j))
    out = []
    for name, scope, fn, thr in COLLISION_RULES:
        pairs = nn if scope == 'nn' else nnn
        for i, j in pairs:
            d = fn(f[i], f[j], a[i], a[j])
            if d < thr:
                out.append({'rule': name, 'pair': (i, j), 'distance_MHz': d * 1000.0, 'threshold_MHz': thr * 1000.0})
    for c in range(dev.n):
        nb = dev.neighbours(c)
        for t in nb:
            for s in nb:
                if s != t:
                    d = abs(f[t] + f[s] - 2 * f[c] - a[c])
                    if d < 0.017:
                        out.append({'rule': 'CR: f_t + f_s = 2 f_c + a_c', 'pair': (c, t, s), 'distance_MHz': d * 1000.0, 'threshold_MHz': 17.0})
    return out

def cr_pairs(dev: 'DeviceConfig') -> List[dict]:
    f = [q.freq for q in dev.qubits]
    a = [q.anharm for q in dev.qubits]
    viol = check_collisions(dev)
    out = []
    for i, j in dev.edges:
        c, t = (i, j) if f[i] > f[j] else (j, i)
        delta = f[c] - f[t]
        bad = [v for v in viol if set(v['pair'][:2]) == {i, j}]
        out.append({'control': c, 'target': t, 'delta_MHz': delta * 1000.0, 'feasible': abs(delta) < abs(a[c]) and (not bad), 'spectators_of_control': [s for s in dev.neighbours(c) if s != t], 'spectators_of_target': [s for s in dev.neighbours(t) if s != c]})
    return out

def make_device(topology: str='chain3', seed: int=0, n_levels: int=3, feedlines: str='single', g: float=0.0035, readout_xtalk: float=0.0, max_tries: int=200, **overrides) -> 'DeviceConfig':
    spec = TOPOLOGIES[topology]
    n, edges = (spec['n'], spec['edges'])
    rng = np.random.default_rng(seed)
    colour = [-1] * n
    colour[0] = 0
    stack = [0]
    while stack:
        k = stack.pop()
        for i, j in edges:
            for u, v in ((i, j), (j, i)):
                if u == k and colour[v] < 0:
                    colour[v] = 1 - colour[k]
                    stack.append(v)
    for attempt in range(max_tries):
        base_hi, base_lo = (5.05, 4.9)
        freqs = []
        for k in range(n):
            base = [5.12, 5.03, 4.92][k] if topology == 'ring3' else base_hi if colour[k] == 0 else base_lo
            freqs.append(base + rng.normal(0, 0.002 if topology == 'ring3' else 0.02))
        qubits = [QubitParams(freq=freqs[k], anharm=-0.33 + rng.normal(0, 0.006), T1=rng.uniform(40000, 90000), Tphi=rng.uniform(60000, 150000), drive_scale=rng.uniform(0.04, 0.06), n_levels=n_levels) for k in range(n)]
        couplings = [CouplingParams(i, j, g=g + rng.normal(0, 0.0003)) for i, j in edges]
        if feedlines == 'single':
            fl = [0] * n
        elif feedlines == 'split':
            fl = [0 if k < (n + 1) // 2 else 1 for k in range(n)]
        else:
            raise ValueError(feedlines)
        resonators = []
        for k in range(n):
            pos = sum((1 for j in range(k) if fl[j] == fl[k]))
            resonators.append(ResonatorParams(freq=7.0 + 0.3 * fl[k] + 0.08 * pos + rng.normal(0, 0.003), kappa=0.0008, chi=-0.0004, feedline=fl[k], snr_scale=1.0, t_int=1000.0))
        rx = None
        if readout_xtalk > 0:
            rx = [[readout_xtalk if j != k and fl[j] == fl[k] else 0.0 for j in range(n)] for k in range(n)]
        dev = DeviceConfig(qubits=qubits, couplings=couplings, resonators=resonators, readout_crosstalk=rx, seed=seed, name=f'{topology}_s{seed}', topology=topology)
        if not check_collisions(dev):
            break
    else:
        raise RuntimeError(f'could not find a collision-free frequency layout for {topology} (seed {seed})')
    for k, v in overrides.items():
        setattr(dev, k, v)
    return dev
