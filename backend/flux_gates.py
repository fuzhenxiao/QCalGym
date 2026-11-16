from copy import deepcopy
import numpy as np
from .flux_coupler import FluxPulse, FluxSegment, finite, KINDS

def validate_gate(dev, value):
    required = {'mode', 'duration', 'amplitudes', 'rise_fractions', 'frequency', 'phase', 'settle', 'frames'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Native flux gate requires exactly ' + ', '.join(sorted(required)))
    if value['mode'] not in ('baseband_cz', 'parametric_cz'):
        raise ValueError('Unsupported native gate')
    finite(value['duration'], 'gate duration', 1.0, 20000.0)
    finite(value['settle'], 'settle', 0.0, 20000.0)
    finite(value['frequency'], 'pump frequency', 0.0, 5.0)
    finite(value['phase'], 'pump phase', -100.0, 100.0)
    for name, size, low, high in [('amplitudes', 3, -0.45, 0.45), ('rise_fractions', 3, 0.0, 0.5), ('frames', 2, -100.0, 100.0)]:
        a = np.asarray(value[name], float)
        if a.shape != (size,) or not np.isfinite(a).all() or np.any(a < low) or np.any(a > high):
            raise ValueError('Invalid ' + name)
    if dev.physics['flux_coupler']['kind'] == KINDS[1] and any(value['amplitudes'][:2]):
        raise ValueError('Fixed transmons have no data-qubit Z pulse ports')
    if value['mode'] == 'baseband_cz' and value['frequency'] != 0:
        raise ValueError('Baseband gate frequency must be zero')
    if value['mode'] == 'parametric_cz' and value['frequency'] <= 0:
        raise ValueError('Parametric gate needs a nonzero carrier')
    return deepcopy(value)

def compile_gate(env, *, repetitions=1):
    gate = validate_gate(env.dev, env.cal.native_flux_gate[0])
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('Positive repetition count required')
    pulses = []
    for port, amp in enumerate(gate['amplitudes']):
        if amp:
            pulses.append(FluxPulse(port, amp, gate['duration'], gate['rise_fractions'][port] * gate['duration'], gate['frequency'] if port == 2 else 0.0, gate['phase'] if port == 2 else 0.0))
    if not pulses:
        raise ValueError('Native gate cannot have all zero amplitudes')
    block = [FluxSegment(tuple(pulses), gate['duration'])]
    if gate['settle']:
        block.append(float(gate['settle']))
    return (block * repetitions, np.asarray(gate['frames']) * repetitions)

def compiled(env, kind, edge, frames=None):
    if tuple(edge) != (0, 1):
        raise ValueError('Only edge (0,1) is installed')
    if kind == 'zz_idle':
        return ([1000.0], np.zeros(2))
    if kind != 'cz':
        raise ValueError('This flux-coupler contract installs CZ, not an implicit CNOT')
    return compile_gate(env)
