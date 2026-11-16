import numpy as np
H_OVER_K_GHZ_K = 0.04799243073366221
FIELDS = {'model', 'base_temperature_K', 'conductance_nW_K5', 'power_scale_nW', 'tau', 'noise_frequency_GHz', 'rate', 'dephasing'}

def validate(p):
    from .hardware import _number
    if set(p) != FIELDS or p['model'] != 'electron_phonon':
        raise ValueError('Invalid electron-phonon thermal parameters')
    for name, value in p.items():
        if name != 'model':
            _number(value, name, 0.0 if name in ('rate', 'dephasing') else 1e-12)

def occupation(temperature, frequency):
    x = H_OVER_K_GHZ_K * frequency / np.maximum(temperature, 1e-12)
    return 0.0 if x > 700 else float(1 / np.expm1(x))

def equilibrium(p, amplitude):
    power = p['power_scale_nW'] * abs(amplitude) ** 2
    temperature = (p['base_temperature_K'] ** 5 + power / p['conductance_nW_K5']) ** 0.2
    return (temperature, occupation(temperature, p['noise_frequency_GHz']))

def rhs(p, amplitude, temperature_squared):
    temperature = np.sqrt(max(float(temperature_squared), 1e-20))
    power = p['power_scale_nW'] * abs(amplitude) ** 2
    c0 = 5 * p['conductance_nW_K5'] * p['base_temperature_K'] ** 3 * p['tau']
    return 2 * (power - p['conductance_nW_K5'] * (temperature ** 5 - p['base_temperature_K'] ** 5)) / c0
